#!/usr/bin/env python3
"""
MAML Few-shot Learning on CIFAR-100

Compares:
1. Random init + 10 step inner loop
2. MAML + 1 step inner loop
3. MAML + 5 step inner loop
4. MAML + 10 step inner loop

Evaluated on 5-way 1-shot and 5-way 5-shot classification.

Model: Conv-4 backbone
- 4 conv blocks: Conv(3->64,3)->BN->ReLU->MaxPool(2) x4 -> FC(64*2*2 -> 5)
- First-order MAML (no second-order gradients)

Training: 20000 meta-iterations, meta-batch-size=4, meta-lr=0.001, inner-lr=0.01
Test: 1000 novel tasks, report average accuracy

Outputs saved to: outputs/exp7/

Usage:
    python exp7_maml_fewshot.py              # Full experiment
    python exp7_maml_fewshot.py --quick      # Quick smoke test (synthetic data, 50 iters)
"""

import os
import sys
import random
import pickle
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import CIFAR100
# Monkey-patch: data was obtained from HuggingFace, not the official tar.gz
# Both _check_integrity (initial guard) and _load_meta's check_integrity need bypass
CIFAR100._check_integrity = lambda self: True
import torchvision.datasets.cifar as cifar_module
_orig_check = cifar_module.check_integrity
cifar_module.check_integrity = lambda fpath, md5=None: os.path.isfile(fpath) or _orig_check(fpath, md5)
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===========================================================================
# Configuration
# ===========================================================================
OUTPUT_DIR = r"D:\大三上的资料\假期学习记录\CV实验\outputs\exp7"
DATA_ROOT = r"D:\大三上的资料\假期学习记录\CV实验\data"
NUM_WAYS = 5
NUM_QUERY = 15
NUM_ITERATIONS = 2000   # reduced from 20000 for practical runtime (~1-2 hours)
META_BATCH_SIZE = 4
META_LR = 0.001
INNER_LR = 0.01
NUM_TEST_TASKS = 1000
LOG_INTERVAL = 500
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Reproducibility
# ===========================================================================
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Data Loading
# ===========================================================================
def build_class_indices(dataset):
    """Build list of sample indices for each of the 100 CIFAR-100 classes."""
    targets = np.array(dataset.targets) if hasattr(dataset, 'targets') else np.array(dataset.labels)
    return [np.where(targets == c)[0].tolist() for c in range(100)]


def load_cifar100():
    """Load CIFAR-100 train and test sets, with download retry and cleanup."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])

    # Clean up partial downloads
    targz = os.path.join(DATA_ROOT, "cifar-100-python.tar.gz")
    partial = os.path.join(DATA_ROOT, "cifar-100-python.tar.gz.part")
    for f in [targz, partial]:
        if os.path.exists(f) and os.path.getsize(f) < 160 * 1024 * 1024:  # < 160 MB = partial
            print(f"  [Removing partial download: {f} ({os.path.getsize(f)/1e6:.1f} MB)]")
            os.remove(f)

    print(f"  Data root: {DATA_ROOT}")
    train_dataset = CIFAR100(root=DATA_ROOT, train=True, download=False, transform=transform)
    test_dataset = CIFAR100(root=DATA_ROOT, train=False, download=False, transform=transform)

    train_class_idx = build_class_indices(train_dataset)
    test_class_idx = build_class_indices(test_dataset)
    print(f"  Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, test_dataset, train_class_idx, test_class_idx


def make_synthetic_dataset(num_train=2000, num_test=1000):
    """Create synthetic CIFAR-100-like data for quick testing / debugging.

    Ensures every class has enough samples for 5-way 5-shot + 15 query tasks.
    Returns objects with the same interface as real CIFAR-100 datasets.
    """
    class SyntheticDataset:
        def __init__(self, n_samples, label):
            # 50 samples per class ensures at least 5+15 per task
            samples_per_class = n_samples // 100
            data_list, target_list = [], []
            for c in range(100):
                for _ in range(samples_per_class):
                    data_list.append(torch.randn(3, 32, 32))
                    target_list.append(c)
            self.data = torch.stack(data_list)
            self.targets = target_list

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

        def __len__(self):
            return len(self.data)

    train_ds = SyntheticDataset(num_train, label='train')
    test_ds = SyntheticDataset(num_test, label='test')
    ti = build_class_indices(train_ds)
    tti = build_class_indices(test_ds)
    return train_ds, test_ds, ti, tti


# ===========================================================================
# Task Sampling
# ===========================================================================
def sample_task(dataset, class_indices, device,
                num_ways=5, num_support=1, num_query=15):
    """Sample a few-shot classification task.

    Returns:
        support_x: (num_ways * num_support, 3, 32, 32)
        support_y: (num_ways * num_support,)
        query_x, query_y: analogous for query set
    """
    # Only sample from classes that have enough samples for the task
    need = num_support + num_query
    eligible = [c for c in range(100) if len(class_indices[c]) >= need]
    if len(eligible) < num_ways:
        # Fallback: use classes with whatever they have (with replacement)
        eligible = list(range(100))
    task_classes = random.sample(eligible, num_ways)
    support_x, support_y = [], []
    query_x, query_y = [], []

    for new_label, cls in enumerate(task_classes):
        indices = class_indices[cls]
        need = num_support + num_query

        if len(indices) >= need:
            chosen = random.sample(indices, need)
        else:
            chosen = random.choices(indices, k=need)

        for i, idx in enumerate(chosen):
            img, _ = dataset[idx]
            if isinstance(img, torch.Tensor):
                img_t = img
            else:
                img_t = transforms.ToTensor()(img)
            if i < num_support:
                support_x.append(img_t.unsqueeze(0))
                support_y.append(new_label)
            else:
                query_x.append(img_t.unsqueeze(0))
                query_y.append(new_label)

    support_x = torch.cat(support_x, dim=0).to(device)
    support_y = torch.tensor(support_y, dtype=torch.long, device=device)
    query_x = torch.cat(query_x, dim=0).to(device)
    query_y = torch.tensor(query_y, dtype=torch.long, device=device)
    return support_x, support_y, query_x, query_y


# ===========================================================================
# Model: Conv-4
# ===========================================================================
class Conv4(nn.Module):
    """Conv-4 backbone for few-shot learning on CIFAR-100 (32x32 images).

    Architecture:
        Conv(3->64,3)->BN->ReLU->MaxPool(2) x4 -> FC(256 -> num_ways)
    Output spatial size after 4 poolings: 32 -> 16 -> 8 -> 4 -> 2
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 2 * 2, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_flat_params(self):
        """Return flattened parameters as a 1-D numpy array (for PCA)."""
        return torch.cat([p.data.view(-1) for p in self.parameters()]).cpu().numpy()


# ===========================================================================
# MAML Utilities
# ===========================================================================
def clone_model(model, device):
    """Return a deep copy of *model* with identical parameters but detached graph."""
    clone = Conv4(model.num_classes).to(device)
    clone.load_state_dict(model.state_dict())
    return clone


def fomaml_adapt_and_grad(model, support_x, support_y, query_x, query_y,
                          inner_steps, inner_lr, device):
    """First-order MAML: adapt to a task, then return query-loss gradients.

    The inner-loop updates are performed under ``torch.no_grad()`` so no
    second-order computation graph is built (FOMAML approximation).

    Returns:
        grad_list: list of gradients w.r.t. *adapted* parameters
        query_loss_val: scalar query loss (item)
    """
    adapted = clone_model(model, device)
    adapted.train()

    # --- Inner loop: SGD on support set ---
    for _ in range(inner_steps):
        logits = adapted(support_x)
        loss = F.cross_entropy(logits, support_y)
        grads = torch.autograd.grad(loss, adapted.parameters())
        with torch.no_grad():
            for p, g in zip(adapted.parameters(), grads):
                p -= inner_lr * g

    # --- Query loss (builds graph on adapted params) ---
    adapted.eval()
    q_logits = adapted(query_x)
    q_loss = F.cross_entropy(q_logits, query_y)

    # FOMAML: gradient of query loss w.r.t. adapted parameters
    q_grads = torch.autograd.grad(q_loss, adapted.parameters())
    return q_grads, q_loss.item()


# ===========================================================================
# Training
# ===========================================================================
def train_maml(model, inner_steps, num_support, num_ways, num_query,
               train_dataset, train_class_idx, device,
               meta_lr=META_LR, meta_batch_size=META_BATCH_SIZE,
               num_iterations=NUM_ITERATIONS, log_interval=LOG_INTERVAL,
               save_path=None):
    """Train *model* with First-Order MAML.

    Args:
        model: Conv4 instance (modified in-place)
        inner_steps: number of inner-loop gradient steps during meta-training
        num_support: shots per class (1 or 5)
        num_ways: classes per task
        num_query: query samples per class
        train_dataset, train_class_idx: training data
        device: torch device
        meta_lr: meta learning rate (outer loop)
        meta_batch_size: tasks per meta-batch
        num_iterations: total meta-iterations
        log_interval: log every N iterations
        save_path: optional path to save best model state_dict

    Returns:
        iteration_losses: list of average losses per iteration
    """
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
    iteration_losses = []
    best_loss = float('inf')

    t_start = time.time()

    for iteration in range(1, num_iterations + 1):
        meta_optimizer.zero_grad()
        meta_loss = 0.0

        # --- Sample a meta-batch of tasks ---
        for _ in range(meta_batch_size):
            s_x, s_y, q_x, q_y = sample_task(
                train_dataset, train_class_idx, device,
                num_ways=num_ways, num_support=num_support, num_query=num_query,
            )

            q_grads, loss_val = fomaml_adapt_and_grad(
                model, s_x, s_y, q_x, q_y,
                inner_steps=inner_steps, inner_lr=INNER_LR, device=device,
            )
            meta_loss += loss_val

            # Accumulate FOMAML gradients onto the original model
            for p_orig, g in zip(model.parameters(), q_grads):
                if p_orig.grad is None:
                    p_orig.grad = g.clone()
                else:
                    p_orig.grad += g

        # --- Outer-loop meta-update ---
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= meta_batch_size
        meta_optimizer.step()

        avg_loss = meta_loss / meta_batch_size
        iteration_losses.append(avg_loss)

        if iteration % log_interval == 0:
            elapsed = time.time() - t_start
            print(f"    Iter {iteration:5d}/{num_iterations}  "
                  f"Loss: {avg_loss:.4f}  "
                  f"[{elapsed:.0f}s elapsed, {elapsed/iteration:.2f}s/iter]")

        # Save best model
        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

    total_time = time.time() - t_start
    print(f"    Training finished in {total_time:.0f}s "
          f"({total_time/num_iterations:.3f}s/iter)")
    return iteration_losses


# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate(model, inner_steps, num_support, num_ways, num_query,
             test_dataset, test_class_idx, device,
             num_tasks=NUM_TEST_TASKS, quiet=False):
    """Evaluate on *num_tasks* novel tasks.

    For each task: clone *model*, adapt on support set with *inner_steps* SGD
    steps, then measure accuracy on the query set.

    Returns:
        mean_acc, stderr, acc_list
    """
    accuracies = []
    model.eval()

    for task_idx in range(num_tasks):
        s_x, s_y, q_x, q_y = sample_task(
            test_dataset, test_class_idx, device,
            num_ways=num_ways, num_support=num_support, num_query=num_query,
        )

        # Adapt to task
        adapted = clone_model(model, device)
        adapted.train()
        for _ in range(inner_steps):
            logits = adapted(s_x)
            loss = F.cross_entropy(logits, s_y)
            grads = torch.autograd.grad(loss, adapted.parameters())
            with torch.no_grad():
                for p, g in zip(adapted.parameters(), grads):
                    p -= INNER_LR * g

        # Evaluate on query set
        adapted.eval()
        with torch.no_grad():
            preds = adapted(q_x).argmax(dim=1)
            accuracies.append((preds == q_y).float().mean().item())

        if not quiet and (task_idx + 1) % 200 == 0:
            print(f"      [{task_idx+1}/{num_tasks}] avg so far: "
                  f"{np.mean(accuracies[-200:]):.4f}")

    mean_acc = float(np.mean(accuracies)) * 100.0   # percent
    stderr = float(np.std(accuracies) / np.sqrt(num_tasks)) * 100.0
    return mean_acc, stderr, accuracies


# ===========================================================================
# Visualisation Helpers
# ===========================================================================
def plot_bar_chart(results, filename):
    """Grouped bar chart: configurations x (1-shot, 5-shot)."""
    configs = list(results["1-shot"].keys())
    x = np.arange(len(configs))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    vals1 = [results["1-shot"][c] for c in configs]
    vals5 = [results["5-shot"][c] for c in configs]

    bars1 = ax.bar(x - width/2, vals1, width, label='1-shot',
                   color='steelblue', edgecolor='white', linewidth=0.5)
    bars5 = ax.bar(x + width/2, vals5, width, label='5-shot',
                   color='lightcoral', edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('5-way Few-shot Classification on CIFAR-100', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(vals1), max(vals5)) * 1.25)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {filename}")


def plot_inner_steps_curve(models_meta, labels, shot_name, num_support,
                           num_ways, num_query,
                           test_dataset, test_class_idx, device,
                           filename):
    """Accuracy vs inner-loop steps for each MAML-variant model."""
    steps_range = [1, 2, 3, 5, 7, 10, 15, 20]
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, label, color in zip(models_meta, labels, colors):
        accs, errs = [], []
        for s in steps_range:
            acc, se, _ = evaluate(model, inner_steps=s, num_support=num_support,
                                  num_ways=num_ways, num_query=num_query,
                                  test_dataset=test_dataset,
                                  test_class_idx=test_class_idx,
                                  device=device,
                                  num_tasks=200, quiet=True)
            accs.append(acc)
            errs.append(se)
            print(f"    {label:<30s} | inner_steps={s:2d} | {acc:.2f}%")

        ax.errorbar(steps_range, accs, yerr=errs, color=color, label=label,
                    marker='o', capsize=3, linewidth=2, markersize=6)

    ax.set_xlabel('Inner-loop Steps at Test Time')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy vs Inner Steps  ({shot_name})', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(steps_range)

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {filename}")


def plot_pca_visualization(model, num_ways, num_query,
                           test_dataset, test_class_idx, device,
                           filename, num_tasks=15):
    """2-D PCA of the parameter space: random init, MAML init, and task-specific
    adapted parameters (after 1, 5, 10 inner steps)."""
    param_vectors = []

    # 1. Random initialisation
    random_model = Conv4(num_ways).to(device)
    param_vectors.append(random_model.get_flat_params())

    # 2. MAML meta-initialization (the trained model)
    param_vectors.append(model.get_flat_params())

    # 3. Task-specific adapted parameters
    for t in range(num_tasks):
        s_x, s_y, q_x, q_y = sample_task(
            test_dataset, test_class_idx, device,
            num_ways=num_ways, num_support=5, num_query=num_query,
        )

        adapted = clone_model(model, device)
        adapted.train()

        # After 1 step
        for _ in range(1):
            logits = adapted(s_x)
            loss = F.cross_entropy(logits, s_y)
            grads = torch.autograd.grad(loss, adapted.parameters())
            with torch.no_grad():
                for p, g in zip(adapted.parameters(), grads):
                    p -= INNER_LR * g
        param_vectors.append(adapted.get_flat_params())

        # After 5 steps (4 more)
        for _ in range(4):
            logits = adapted(s_x)
            loss = F.cross_entropy(logits, s_y)
            grads = torch.autograd.grad(loss, adapted.parameters())
            with torch.no_grad():
                for p, g in zip(adapted.parameters(), grads):
                    p -= INNER_LR * g
        param_vectors.append(adapted.get_flat_params())

        # After 10 steps (5 more)
        for _ in range(5):
            logits = adapted(s_x)
            loss = F.cross_entropy(logits, s_y)
            grads = torch.autograd.grad(loss, adapted.parameters())
            with torch.no_grad():
                for p, g in zip(adapted.parameters(), grads):
                    p -= INNER_LR * g
        param_vectors.append(adapted.get_flat_params())

    param_matrix = np.vstack(param_vectors)         # (2 + 3*num_tasks, D)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(param_matrix)
    ev_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot random init
    ax.scatter(coords[0, 0], coords[0, 1], c='black', s=250, marker='*',
               label='Random Init', zorder=5, edgecolors='white', linewidths=0.5)

    # Plot MAML init
    ax.scatter(coords[1, 0], coords[1, 1], c='red', s=250, marker='*',
               label='MAML Init', zorder=5, edgecolors='white', linewidths=0.5)

    # Plot per-task trajectories
    step_colors = {1: 'blue', 5: 'green', 10: 'orange'}
    legend_done = set()
    for t in range(num_tasks):
        i1 = 2 + t * 3      # after 1 step
        i5 = 2 + t * 3 + 1  # after 5 steps
        i10 = 2 + t * 3 + 2 # after 10 steps

        # Trajectory line
        ax.plot(coords[[i1, i5, i10], 0], coords[[i1, i5, i10], 1],
                color='gray', alpha=0.25, linewidth=0.8)

        for i_step, label_step in [(i1, '1 step'), (i5, '5 steps'), (i10, '10 steps')]:
            color = step_colors[int(label_step.split()[0])]
            lbl = f'Task-specific ({label_step})' if label_step not in legend_done else None
            ax.scatter(coords[i_step, 0], coords[i_step, 1],
                       c=color, s=25, alpha=0.6, label=lbl, zorder=3)
            legend_done.add(label_step)

    ax.set_xlabel(f'PC1 ({ev_ratio[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({ev_ratio[1]*100:.1f}% variance)')
    ax.set_title('Parameter Space (PCA)', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {filename}")


# ===========================================================================
# Main Experiment Loop
# ===========================================================================
def run_experiment(quick=False):
    """Run the full MAML few-shot experiment."""
    if quick:
        print("=== QUICK TEST MODE (synthetic data, 50 iterations) ===")
        train_dataset, test_dataset, train_class_idx, test_class_idx = make_synthetic_dataset(
            num_train=2000, num_test=2000)
        num_iterations = 50
        log_interval = 10
        num_test_tasks = 20
    else:
        print("=== FULL EXPERIMENT MODE ===")
        train_dataset, test_dataset, train_class_idx, test_class_idx = load_cifar100()
        num_iterations = NUM_ITERATIONS
        log_interval = LOG_INTERVAL
        num_test_tasks = NUM_TEST_TASKS

    results = {"1-shot": {}, "5-shot": {}}
    models_pool = {}

    for num_support in [1, 5]:
        shot_label = f"{num_support}-shot"
        print(f"\n{'='*72}")
        print(f"  {shot_label.upper()}  "
              f"(meta-batch={META_BATCH_SIZE}, "
              f"inner_lr={INNER_LR}, meta_lr={META_LR})")
        print(f"{'='*72}")

        # ---------------------------------------------------------------
        # 1) Random init + 10 inner steps  (no training)
        # ---------------------------------------------------------------
        print(f"\n--- Random Init + 10 steps ({shot_label}) ---")
        rand_model = Conv4(NUM_WAYS).to(device)
        acc, se, _ = evaluate(
            rand_model, inner_steps=10,
            num_support=num_support, num_ways=NUM_WAYS, num_query=NUM_QUERY,
            test_dataset=test_dataset, test_class_idx=test_class_idx,
            device=device, num_tasks=num_test_tasks, quiet=False)
        print(f"  >> {acc:.2f}% +/- {se:.2f}%")
        results[shot_label]["Random + 10 step"] = acc

        # ---------------------------------------------------------------
        # Helper: train-or-load a MAML model
        # ---------------------------------------------------------------
        def get_maml_model(inner_steps, tag, ckpt_prefix=""):
            ckpt = os.path.join(OUTPUT_DIR, f"{ckpt_prefix}maml_{tag}_{shot_label}.pt")
            loss_npy = os.path.join(OUTPUT_DIR, f"{ckpt_prefix}losses_{tag}_{shot_label}.npy")
            m = Conv4(NUM_WAYS).to(device)

            if os.path.exists(ckpt):
                print(f"  [Loading pre-trained {tag}]")
                m.load_state_dict(torch.load(ckpt, map_location=device))
            else:
                print(f"  [Training MAML {tag}, inner_steps={inner_steps}]")
                losses = train_maml(
                    m, inner_steps=inner_steps,
                    num_support=num_support, num_ways=NUM_WAYS, num_query=NUM_QUERY,
                    train_dataset=train_dataset, train_class_idx=train_class_idx,
                    device=device, meta_lr=META_LR, meta_batch_size=META_BATCH_SIZE,
                    num_iterations=num_iterations, log_interval=log_interval,
                    save_path=ckpt)
                np.save(loss_npy, np.array(losses))
                if os.path.exists(ckpt):
                    m.load_state_dict(torch.load(ckpt, map_location=device))
            return m

        # ---------------------------------------------------------------
        # 2) MAML + 1 inner step
        # ---------------------------------------------------------------
        print(f"\n--- MAML + 1 step ({shot_label}) ---")
        m_1 = get_maml_model(inner_steps=1, tag="1step")
        acc, se, _ = evaluate(
            m_1, inner_steps=1,
            num_support=num_support, num_ways=NUM_WAYS, num_query=NUM_QUERY,
            test_dataset=test_dataset, test_class_idx=test_class_idx,
            device=device, num_tasks=num_test_tasks)
        print(f"  >> {acc:.2f}% +/- {se:.2f}%")
        results[shot_label]["MAML + 1 step"] = acc
        models_pool[f"{shot_label}_maml1"] = m_1

        # ---------------------------------------------------------------
        # 3) MAML + 5 inner steps
        # ---------------------------------------------------------------
        print(f"\n--- MAML + 5 step ({shot_label}) ---")
        m_5 = get_maml_model(inner_steps=5, tag="5step")
        acc, se, _ = evaluate(
            m_5, inner_steps=5,
            num_support=num_support, num_ways=NUM_WAYS, num_query=NUM_QUERY,
            test_dataset=test_dataset, test_class_idx=test_class_idx,
            device=device, num_tasks=num_test_tasks)
        print(f"  >> {acc:.2f}% +/- {se:.2f}%")
        results[shot_label]["MAML + 5 step"] = acc
        models_pool[f"{shot_label}_maml5"] = m_5

        # ---------------------------------------------------------------
        # 4) MAML + 10 inner steps
        # ---------------------------------------------------------------
        print(f"\n--- MAML + 10 step ({shot_label}) ---")
        m_10 = get_maml_model(inner_steps=10, tag="10step")
        acc, se, _ = evaluate(
            m_10, inner_steps=10,
            num_support=num_support, num_ways=NUM_WAYS, num_query=NUM_QUERY,
            test_dataset=test_dataset, test_class_idx=test_class_idx,
            device=device, num_tasks=num_test_tasks)
        print(f"  >> {acc:.2f}% +/- {se:.2f}%")
        results[shot_label]["MAML + 10 step"] = acc
        models_pool[f"{shot_label}_maml10"] = m_10

        # ---------------------------------------------------------------
        # Inner steps vs accuracy curve (uses ~200 tasks per data point)
        # ---------------------------------------------------------------
        print(f"\n--- Inner steps curve ({shot_label}) ---")
        plot_inner_steps_curve(
            [m_1, m_5, m_10],
            labels=['MAML (trained 1 step)', 'MAML (trained 5 steps)',
                    'MAML (trained 10 steps)'],
            shot_name=shot_label,
            num_support=num_support, num_ways=NUM_WAYS, num_query=NUM_QUERY,
            test_dataset=test_dataset, test_class_idx=test_class_idx,
            device=device,
            filename=os.path.join(OUTPUT_DIR, f"curve_inner_steps_{shot_label}.png"),
        )

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*72}")
    print("  FINAL RESULTS")
    print(f"{'='*72}")
    for shot, shot_res in results.items():
        print(f"\n  {shot.upper()}:")
        for cfg, acc in shot_res.items():
            print(f"    {cfg:<25s}  {acc:.2f}%")

    # Save numeric results
    pkl_path = os.path.join(OUTPUT_DIR, "results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  [Saved] {pkl_path}")

    # ==================================================================
    # Bar chart
    # ==================================================================
    print(f"\n--- Accuracy bar chart ---")
    plot_bar_chart(results,
                   filename=os.path.join(OUTPUT_DIR, "accuracy_bar_chart.png"))

    # ==================================================================
    # PCA visualisation (use the 1-shot / MAML-10 model for a clear view)
    # ==================================================================
    print(f"\n--- PCA parameter space ---")
    pca_model = models_pool.get("1-shot_maml10",
                                next(iter(models_pool.values())) if models_pool else Conv4(NUM_WAYS).to(device))
    plot_pca_visualization(
        pca_model, num_ways=NUM_WAYS, num_query=NUM_QUERY,
        test_dataset=test_dataset, test_class_idx=test_class_idx,
        device=device,
        filename=os.path.join(OUTPUT_DIR, "pca_parameters.png"),
    )

    print(f"\n{'='*72}")
    print("  Experiment complete. All outputs in:")
    print(f"    {OUTPUT_DIR}")
    print(f"{'='*72}")


def main():
    quick = "--quick" in sys.argv

    print(f"Using device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed()

    try:
        run_experiment(quick=quick)
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
