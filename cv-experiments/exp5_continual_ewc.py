#!/usr/bin/env python3
"""
Experiment 5: Continual Learning with Elastic Weight Consolidation (EWC) on Split CIFAR-100.

Split CIFAR-100 (100 classes) into 5 disjoint tasks of 20 classes each.
Compare: Fine-tune, EWC (λ = 100, 1000, 10000), Rehearsal (2000 samples/old task), Joint.

Model: Custom ResNet-32 for CIFAR (~0.47M params, 3 stages with 5 residual blocks each).
"""

import os, sys, copy, json, time, warnings, random, pickle as pk
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

# ──────── Performance ──────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ──────────────────── Monkey-patch torchvision integrity checks ────────────
# Our CIFAR-100 data was built from HuggingFace (not the original CS Toronto tar.gz),
# so MD5 checksums don't match.
torchvision.datasets.cifar.CIFAR10._check_integrity = lambda self: True
torchvision.datasets.cifar.CIFAR100._check_integrity = lambda self: True
def _patched_load_meta(self):
    path = os.path.join(self.root, self.base_folder, self.meta["filename"])
    if not os.path.exists(path):
        raise RuntimeError("Dataset metadata not found.")
    with open(path, "rb") as _fp:
        data = pk.load(_fp, encoding="latin1")
        self.classes = data[self.meta["key"]]
    self.class_to_idx = {_c: i for i, _c in enumerate(self.classes)}
torchvision.datasets.cifar.CIFAR10._load_meta = _patched_load_meta
torchvision.datasets.cifar.CIFAR100._load_meta = _patched_load_meta
# ───────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "exp5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_TASKS = 5
CLASSES_PER_TASK = 20
TASK_CLASS_RANGES = [(t * CLASSES_PER_TASK, (t + 1) * CLASSES_PER_TASK)
                     for t in range(NUM_TASKS)]

TRAIN_EPOCHS = 20
BATCH_SIZE = 1024
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
REHEARSAL_BUFFER_SIZE = 2000
EWC_LAMBDAS = [100, 1000, 10000]
NUM_WORKERS = 0
SEED = 42

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────── Reproducibility ──────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════
#  1. ResNet-32 for CIFAR
# ══════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet for CIFAR: num_blocks = (n, n, n) => 6n+2 layers."""

    def __init__(self, block, num_blocks, num_classes=CLASSES_PER_TASK):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet32(num_classes=CLASSES_PER_TASK):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expand_classifier(model, new_num_classes):
    """Expand the linear classifier to accommodate more classes."""
    old = model.linear
    device = old.weight.device
    new = nn.Linear(old.in_features, new_num_classes)
    new.weight.data[:old.out_features] = old.weight.data
    new.bias.data[:old.out_features] = old.bias.data
    nn.init.kaiming_normal_(new.weight[old.out_features:], mode="fan_out",
                            nonlinearity="relu")
    new.bias.data[old.out_features:].zero_()
    model.linear = new.to(device)
    return model


# ══════════════════════════════════════════════════════════════════════
#  2. Data Loading: Split CIFAR-100
# ══════════════════════════════════════════════════════════════════════

def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])


def get_cifar100(data_dir, train=True):
    return torchvision.datasets.CIFAR100(
        root=data_dir, train=train, download=False,
        transform=get_transform(train=train),
    )


def get_task_subset_indices(dataset, task_id):
    c_start, c_end = TASK_CLASS_RANGES[task_id]
    return [i for i, (_, label) in enumerate(dataset) if c_start <= label < c_end]


class TaskSubset(Dataset):
    """Wrapper that extracts a task's classes and remaps labels to [0, K-1]."""

    def __init__(self, dataset, task_id):
        self.dataset = dataset
        self.task_id = task_id
        self.c_start, self.c_end = TASK_CLASS_RANGES[task_id]
        self.indices = get_task_subset_indices(dataset, task_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, label - self.c_start


def get_task_test_loaders(data_dir):
    test_dataset = get_cifar100(data_dir, train=False)
    loaders = []
    for t in range(NUM_TASKS):
        subset = TaskSubset(test_dataset, t)
        loaders.append(DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS))
    return loaders, test_dataset


def get_joint_loaders(data_dir):
    train_ds = get_cifar100(data_dir, train=True)
    test_ds = get_cifar100(data_dir, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)
    return train_loader, test_loader, train_ds


# ══════════════════════════════════════════════════════════════════════
#  3. Training / Evaluation Utilities
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataset_whole, optimizer, criterion, device,
                    ewc_fn=None, batch_size=BATCH_SIZE):
    # Create a fresh DataLoader each epoch to avoid Windows DataLoader iteration hangs
    loader = DataLoader(dataset_whole, batch_size=batch_size, shuffle=True,
                        num_workers=0)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_ce = criterion(outputs, targets)
        loss = loss_ce
        if ewc_fn is not None:
            loss = loss + ewc_fn(model)
        loss.backward()
        optimizer.step()
        running_loss += loss_ce.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100.0 * correct / total


def train_task(model, train_dataset, num_epochs=TRAIN_EPOCHS, device=DEVICE,
               ewc_fn=None, desc="Training"):
    optimizer = optim.SGD(model.parameters(), lr=BASE_LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    losses, accs = [], []
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        loss, acc = train_one_epoch(model, train_dataset, optimizer, criterion,
                                     device, ewc_fn)
        scheduler.step()
        losses.append(loss)
        accs.append(acc)
        dt = time.time() - t0
        print(f"    Epoch {epoch:2d}/{num_epochs}  Loss: {loss:.3f}  Acc: {acc:.2f}%  [{dt:.1f}s]",
              flush=True)
        torch.cuda.empty_cache()
    return losses, accs


@torch.no_grad()
def evaluate(model, loader, device, num_classes=None):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        if num_classes is not None:
            outputs = outputs[:, :num_classes]
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_per_task(model, task_test_loaders, num_tasks, device,
                      global_classes_seen=None):
    model.eval()
    accs = []
    for t in range(num_tasks):
        n_cls = global_classes_seen if global_classes_seen else (t + 1) * CLASSES_PER_TASK
        acc = evaluate(model, task_test_loaders[t], device, num_classes=n_cls)
        accs.append(acc)
    return accs


# ══════════════════════════════════════════════════════════════════════
#  4. EWC Utilities
# ══════════════════════════════════════════════════════════════════════

def is_backbone_param(name):
    return "linear" not in name


def compute_fisher(model, dataset, criterion, device):
    """Diagonal Fisher: F_i = (1/N) * sum(grad_i^2)"""
    # Create a fresh DataLoader each call
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    fisher = {}
    for name, p in model.named_parameters():
        if p.requires_grad and is_backbone_param(name):
            fisher[name] = torch.zeros_like(p.data)
    n = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        bs = inputs.size(0)
        for name, p in model.named_parameters():
            if p.requires_grad and is_backbone_param(name) and p.grad is not None:
                fisher[name] += p.grad.detach() ** 2 * bs
        n += bs
    for name in fisher:
        fisher[name] /= n
    return fisher, n


def ewc_penalty(model, fisher_dict, optimal_params_dict, lambda_ewc, device):
    loss = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in fisher_dict and name in optimal_params_dict:
            if param.shape == optimal_params_dict[name].shape:
                f = fisher_dict[name].to(device)
                theta_star = optimal_params_dict[name].to(device)
                loss = loss + torch.sum(f * (param - theta_star) ** 2)
    return (lambda_ewc / 2.0) * loss


def make_ewc_fn(fisher_dict, optimal_params, lambda_ewc):
    def fn(model):
        return ewc_penalty(model, fisher_dict, optimal_params, lambda_ewc, DEVICE)
    return fn


# ══════════════════════════════════════════════════════════════════════
#  5. Experiment Methods
# ══════════════════════════════════════════════════════════════════════

def run_finetune(train_dataset, task_test_loaders, data_dir):
    print("\n" + "=" * 60, flush=True)
    print("  METHOD: Fine-tune (no protection)", flush=True)
    print("=" * 60, flush=True)
    model = resnet32(num_classes=CLASSES_PER_TASK).to(DEVICE)
    print(f"  Model params: {count_params(model):,}", flush=True)
    acc_matrix = np.zeros((NUM_TASKS, NUM_TASKS))
    task1_acc_over_time = []
    for task_id in range(NUM_TASKS):
        print(f"\n  ─── Task {task_id+1} ───", flush=True)
        if task_id > 0:
            model = expand_classifier(model, (task_id + 1) * CLASSES_PER_TASK)
        task_data = TaskSubset(train_dataset, task_id)
        train_task(model, task_data, desc=f"Fine-tune Task {task_id+1}")
        seen = (task_id + 1) * CLASSES_PER_TASK
        accs = evaluate_per_task(model, task_test_loaders, NUM_TASKS, DEVICE,
                                  global_classes_seen=seen)
        for t in range(task_id + 1):
            acc_matrix[task_id][t] = accs[t]
        task1_acc_over_time.append(accs[0])
        print(f"  -> mean acc = {np.mean(accs[:task_id+1]):.2f}%,  "
              f"Task 1 = {accs[0]:.2f}%", flush=True)
        torch.cuda.empty_cache()
    return model, acc_matrix, task1_acc_over_time


def run_ewc(train_dataset, task_test_loaders, data_dir, lambda_ewc):
    print("\n" + "=" * 60, flush=True)
    print(f"  METHOD: EWC (λ = {lambda_ewc})", flush=True)
    print("=" * 60, flush=True)
    model = resnet32(num_classes=CLASSES_PER_TASK).to(DEVICE)
    print(f"  Model params: {count_params(model):,}", flush=True)
    fisher_dict = {}
    optimal_params = {}
    acc_matrix = np.zeros((NUM_TASKS, NUM_TASKS))
    task1_acc_over_time = []
    fisher_histories = []
    for task_id in range(NUM_TASKS):
        print(f"\n  ─── Task {task_id+1} ───", flush=True)
        if task_id > 0:
            model = expand_classifier(model, (task_id + 1) * CLASSES_PER_TASK)
        task_data = TaskSubset(train_dataset, task_id)
        ewc_fn = None
        if task_id > 0 and len(fisher_dict) > 0:
            ewc_fn = make_ewc_fn(fisher_dict, optimal_params, lambda_ewc)
        train_task(model, task_data, ewc_fn=ewc_fn, desc=f"EWC(λ={lambda_ewc})")
        # Compute Fisher after training
        fisher_dataset = TaskSubset(train_dataset, task_id)
        criterion = nn.CrossEntropyLoss()
        new_fisher, _ = compute_fisher(model, fisher_dataset, criterion, DEVICE)
        if task_id == 0:
            fisher_dict = {n: f.clone() for n, f in new_fisher.items()}
        else:
            for n in fisher_dict:
                if n in new_fisher:
                    fisher_dict[n] = fisher_dict[n] + new_fisher[n]
        optimal_params = {
            n: p.data.clone().cpu()
            for n, p in model.named_parameters()
            if p.requires_grad and is_backbone_param(n)
        }
        fisher_histories.append({
            "task": task_id,
            "fisher": {n: f.mean().item() for n, f in new_fisher.items()},
        })
        seen = (task_id + 1) * CLASSES_PER_TASK
        accs = evaluate_per_task(model, task_test_loaders, NUM_TASKS, DEVICE,
                                  global_classes_seen=seen)
        for t in range(task_id + 1):
            acc_matrix[task_id][t] = accs[t]
        task1_acc_over_time.append(accs[0])
        print(f"  -> mean acc = {np.mean(accs[:task_id+1]):.2f}%,  "
              f"Task 1 = {accs[0]:.2f}%", flush=True)
        torch.cuda.empty_cache()
    return model, acc_matrix, task1_acc_over_time, fisher_histories


def run_rehearsal(train_dataset, task_test_loaders, data_dir):
    print("\n" + "=" * 60, flush=True)
    print(f"  METHOD: Rehearsal ({REHEARSAL_BUFFER_SIZE} samples/old task)", flush=True)
    print("=" * 60, flush=True)
    model = resnet32(num_classes=CLASSES_PER_TASK).to(DEVICE)
    print(f"  Model params: {count_params(model):,}", flush=True)
    buffer_datasets = []
    acc_matrix = np.zeros((NUM_TASKS, NUM_TASKS))
    task1_acc_over_time = []
    for task_id in range(NUM_TASKS):
        print(f"\n  ─── Task {task_id+1} ───", flush=True)
        if task_id > 0:
            model = expand_classifier(model, (task_id + 1) * CLASSES_PER_TASK)
        task_data = TaskSubset(train_dataset, task_id)
        if task_id == 0:
            train_task(model, task_data, desc=f"Rehearsal Task {task_id+1}")
        else:
            combined = ConcatDataset([task_data] + buffer_datasets)
            train_task(model, combined, desc=f"Rehearsal Task {task_id+1}")
        # Sample buffer
        idxs = np.random.choice(
            len(task_data),
            size=min(REHEARSAL_BUFFER_SIZE, len(task_data)),
            replace=False,
        )
        buffer_datasets.append(Subset(task_data, idxs))
        seen = (task_id + 1) * CLASSES_PER_TASK
        accs = evaluate_per_task(model, task_test_loaders, NUM_TASKS, DEVICE,
                                  global_classes_seen=seen)
        for t in range(task_id + 1):
            acc_matrix[task_id][t] = accs[t]
        task1_acc_over_time.append(accs[0])
        print(f"  -> mean acc = {np.mean(accs[:task_id+1]):.2f}%,  "
              f"Task 1 = {accs[0]:.2f}%", flush=True)
        torch.cuda.empty_cache()
    return model, acc_matrix, task1_acc_over_time


def run_joint(data_dir, task_test_loaders):
    print("\n" + "=" * 60, flush=True)
    print("  METHOD: Joint training (upper bound)", flush=True)
    print("=" * 60, flush=True)
    model = resnet32(num_classes=NUM_TASKS * CLASSES_PER_TASK).to(DEVICE)
    print(f"  Model params: {count_params(model):,}", flush=True)
    train_loader, test_loader, joint_train_ds = get_joint_loaders(data_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=BASE_LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        # Create fresh loader each epoch
        train_loader = DataLoader(joint_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        scheduler.step()
        dt = time.time() - t0
        if True:
            print(f"    Epoch {epoch:2d}/{TRAIN_EPOCHS}  Loss: {running_loss/total:.3f}  "
                  f"Acc: {100.*correct/total:.2f}%  [{dt:.1f}s]", flush=True)
    per_task_accs = evaluate_per_task(model, task_test_loaders, NUM_TASKS, DEVICE,
                                       global_classes_seen=100)
    acc_matrix = np.zeros((NUM_TASKS, NUM_TASKS))
    for t in range(NUM_TASKS):
        for t2 in range(NUM_TASKS):
            acc_matrix[t][t2] = per_task_accs[t2]
    print(f"  -> per-task accs: {[f'{a:.2f}%' for a in per_task_accs]}", flush=True)
    print(f"  -> mean: {np.mean(per_task_accs):.2f}%", flush=True)
    return model, acc_matrix, per_task_accs


# ══════════════════════════════════════════════════════════════════════
#  6. Visualization & Saving
# ══════════════════════════════════════════════════════════════════════

def save_accuracy_matrix(acc_matrix, method_name, save_dir):
    path_csv = os.path.join(save_dir, f"accuracy_matrix_{method_name}.csv")
    try:
        np.savetxt(path_csv, acc_matrix, fmt="%.2f", delimiter=",",
                   header=",".join([f"Task{t+1}" for t in range(NUM_TASKS)]))
    except FileNotFoundError:
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(path_csv, acc_matrix, fmt="%.2f", delimiter=",",
                   header=",".join([f"Task{t+1}" for t in range(NUM_TASKS)]))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(acc_matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(NUM_TASKS))
    ax.set_yticks(range(NUM_TASKS))
    ax.set_xticklabels([f"Task {t+1}" for t in range(NUM_TASKS)])
    ax.set_yticklabels([f"After\nTask {t+1}" for t in range(NUM_TASKS)])
    for i in range(NUM_TASKS):
        for j in range(NUM_TASKS):
            if acc_matrix[i][j] > 0:
                c = "white" if acc_matrix[i][j] > 50 else "black"
                ax.text(j, i, f"{acc_matrix[i][j]:.1f}", ha="center", va="center",
                        fontsize=9, color=c)
    plt.title(f"Accuracy Matrix — {method_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"accuracy_matrix_{method_name}.png"), dpi=150)
    plt.close()


def plot_task1_curve(all_results, save_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "D", "^", "v", "p"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for idx, (method_name, task1_accs) in enumerate(all_results.items()):
        tasks = list(range(1, len(task1_accs) + 1))
        ax.plot(tasks, task1_accs, marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)], linewidth=2, markersize=8,
                label=method_name)
    ax.set_xlabel("After Training on Task", fontsize=13)
    ax.set_ylabel("Task 1 Accuracy (%)", fontsize=13)
    ax.set_title("Forgetting Curve: Task 1 Accuracy Over Time", fontsize=14)
    ax.set_xticks(range(1, NUM_TASKS + 1))
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "task1_accuracy_curve.png"), dpi=150)
    plt.close()


def plot_fisher_heatmap(fisher_histories, save_dir, top_k=50):
    if len(fisher_histories) < 2:
        print("  Skipping Fisher heatmap (need >=2 tasks)", flush=True)
        return
    all_names = sorted(set().union(*(h["fisher"].keys() for h in fisher_histories)))
    nt = len(fisher_histories)
    mat = np.zeros((nt, len(all_names)))
    for t, h in enumerate(fisher_histories):
        for pidx, pname in enumerate(all_names):
            mat[t, pidx] = h["fisher"].get(pname, 0.0)
    top = np.argsort(mat.mean(axis=0))[-top_k:]
    top_data = mat[:, top]
    short = [n.replace(".weight", "").replace(".bias", "") for n in
             [all_names[i] for i in top]]
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(top_data, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(nt))
    ax.set_yticklabels([f"Task {t+1}" for t in range(nt)])
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    plt.colorbar(im, ax=ax, label="Fisher Information (mean per-param)")
    ax.set_title(f"Top-{top_k} Parameters by Fisher Information", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fisher_heatmap.png"), dpi=150)
    plt.close()
    print(f"  Saved fisher_heatmap.png", flush=True)


def save_lambda_ablation(all_summaries, save_dir):
    import csv
    rows = []
    for e in all_summaries:
        rows.append({
            "Method": e["method"],
            "Final Avg Acc (%)": f"{e['final_avg']:.2f}",
            "Task 1 Acc After All (%)": f"{e['task1_final']:.2f}",
            "Forgetting (Task 1 drop, pp)": f"{e['forgetting']:.2f}",
        })
    path = os.path.join(save_dir, "lambda_ablation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Ablation table:", flush=True)
    print(f"  {'Method':25s} {'Final Avg':>10s} {'T1 After All':>14s} {'Forgetting':>12s}", flush=True)
    print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*12}", flush=True)
    for r in rows:
        print(f"  {r['Method']:25s} {r['Final Avg Acc (%)']:>10s} "
              f"{r['Task 1 Acc After All (%)']:>14s} {r['Forgetting (Task 1 drop, pp)']:>12s}",
              flush=True)
    return rows


def save_results_json(all_summaries, save_dir):
    import json
    path = os.path.join(save_dir, "all_results.json")
    with open(path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"  Saved {path}", flush=True)


def plot_combined_accuracy_matrices(results_dict, save_dir):
    n = len(results_dict)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, (method_name, info) in enumerate(results_dict.items()):
        if idx >= 6:
            break
        m = info["acc_matrix"]
        ax = axes[idx]
        im = ax.imshow(m, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set_xticks(range(NUM_TASKS))
        ax.set_yticks(range(NUM_TASKS))
        ax.set_xticklabels([f"T{t+1}" for t in range(NUM_TASKS)])
        ax.set_yticklabels([f"After\nT{t+1}" for t in range(NUM_TASKS)])
        for i in range(NUM_TASKS):
            for j in range(NUM_TASKS):
                if m[i][j] > 0:
                    c = "white" if m[i][j] > 50 else "black"
                    ax.text(j, i, f"{m[i][j]:.0f}", ha="center", va="center",
                            fontsize=8, color=c, fontweight="bold")
        ax.set_title(method_name, fontsize=12, fontweight="bold")
    for idx in range(n, 6):
        axes[idx].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_accuracy_matrices.png"), dpi=150)
    plt.close()
    print(f"  Saved all_accuracy_matrices.png", flush=True)


def compute_summary(method_name, acc_matrix, task1_acc_over_time):
    final_avg = np.mean([acc_matrix[NUM_TASKS - 1][t] for t in range(NUM_TASKS)])
    t1_final = task1_acc_over_time[-1] if len(task1_acc_over_time) > 0 else 0.0
    t1_init = task1_acc_over_time[0] if len(task1_acc_over_time) > 0 else 0.0
    return {"method": method_name, "final_avg": final_avg,
            "task1_final": t1_final, "forgetting": t1_init - t1_final}


# ══════════════════════════════════════════════════════════════════════
#  7. Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60, flush=True)
    print("  Experiment 5: Continual Learning on Split CIFAR-100", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print("=" * 60, flush=True)

    data_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    print("\n  Loading CIFAR-100 ...", flush=True)
    train_dataset = get_cifar100(data_dir, train=True)
    print(f"    Train: {len(train_dataset)} samples", flush=True)

    task_test_loaders, _ = get_task_test_loaders(data_dir)
    for t in range(NUM_TASKS):
        n = len(get_task_subset_indices(train_dataset, t))
        r = TASK_CLASS_RANGES[t]
        print(f"    Task {t+1}: classes {r[0]}-{r[1]-1}, {n} train / {len(task_test_loaders[t].dataset)} test",
              flush=True)

    results = {}

    # 1. Fine-tune
    set_seed()
    _, acc_ft, t1_ft = run_finetune(train_dataset, task_test_loaders, data_dir)
    results["Fine-tune"] = {"acc_matrix": acc_ft, "task1_accs": t1_ft}
    save_accuracy_matrix(acc_ft, "finetune", OUTPUT_DIR)

    # 2. EWC
    ewc_all_fisher = {}
    for lam in EWC_LAMBDAS:
        set_seed()
        _, acc_ewc, t1_ewc, fisher_hist = run_ewc(
            train_dataset, task_test_loaders, data_dir, lam)
        label = f"EWC λ={lam}"
        results[label] = {"acc_matrix": acc_ewc, "task1_accs": t1_ewc}
        save_accuracy_matrix(acc_ewc, f"ewc_lambda_{lam}", OUTPUT_DIR)
        if lam == 1000:
            ewc_all_fisher = fisher_hist

    # 3. Rehearsal
    set_seed()
    _, acc_reh, t1_reh = run_rehearsal(train_dataset, task_test_loaders, data_dir)
    results["Rehearsal"] = {"acc_matrix": acc_reh, "task1_accs": t1_reh}
    save_accuracy_matrix(acc_reh, "rehearsal", OUTPUT_DIR)

    # 4. Joint
    set_seed()
    _, acc_joint, t1_joint = run_joint(data_dir, task_test_loaders)
    results["Joint"] = {"acc_matrix": acc_joint, "task1_accs": t1_joint}
    save_accuracy_matrix(acc_joint, "joint", OUTPUT_DIR)

    # ── Combined plots ───────────────────────────────────────
    plot_combined_accuracy_matrices(results, OUTPUT_DIR)

    t1_curves = {}
    for method_name, info in results.items():
        t1 = info["task1_accs"]
        arr = np.atleast_1d(np.asarray(t1, dtype=float))
        if len(arr) == 1:
            arr = np.full(NUM_TASKS, arr[0])
        t1_curves[method_name] = arr.tolist()
    plot_task1_curve(t1_curves, OUTPUT_DIR)

    if len(ewc_all_fisher) >= 2:
        plot_fisher_heatmap(ewc_all_fisher, OUTPUT_DIR, top_k=50)

    # ── Summary table ─────────────────────────────────────────
    all_summaries = []
    all_summaries.append(compute_summary("Fine-tune", acc_ft, t1_ft))
    for lam in EWC_LAMBDAS:
        label = f"EWC λ={lam}"
        info = results[label]
        all_summaries.append(compute_summary(label, info["acc_matrix"], info["task1_accs"]))
    all_summaries.append(compute_summary("Rehearsal", acc_reh, t1_reh))
    j_final = np.mean([acc_joint[NUM_TASKS - 1][t] for t in range(NUM_TASKS)])
    j_t1 = float(np.atleast_1d(np.asarray(t1_joint, dtype=float))[0])
    all_summaries.append({"method": "Joint (upper bound)", "final_avg": j_final,
                          "task1_final": j_t1, "forgetting": 0.0})

    save_lambda_ablation(all_summaries, OUTPUT_DIR)
    save_results_json(all_summaries, OUTPUT_DIR)

    print("\n" + "=" * 60, flush=True)
    print("  ALL EXPERIMENTS COMPLETE", flush=True)
    print(f"  Results in: {OUTPUT_DIR}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
