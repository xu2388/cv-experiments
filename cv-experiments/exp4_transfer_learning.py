#!/usr/bin/env python3
"""
Transfer Learning Experiment on CIFAR-100 with ResNet-18.
Compares 4 strategies:
  1. Scratch          – train from random init
  2. Linear Probe     – freeze backbone, train classifier head only
  3. Full Fine-tune   – train all parameters from pretrained
  4. LoRA             – low-rank adapters on all Conv2d layers + head

Target domain: 20 random classes from CIFAR-100, 20 train + 50 test per class.

If the real CIFAR-100 download from torchvision fails (slow network), the script
automatically falls back to synthetic data so the pipeline is verifiable end-to-end.
"""

import os
import sys
import random
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =====================================================================
#  Configuration
# =====================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "exp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 20
NUM_TRAIN_PER_CLASS = 20
NUM_TEST_PER_CLASS = 50
EPOCHS = 30
BATCH_SIZE = 64

# CIFAR-100 dataset-wide stats
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)

# =====================================================================
#  Synthetic CIFAR-100 fallback
# =====================================================================


class SyntheticCIFAR100(Dataset):
    """Procedurally generated CIFAR-100-like dataset.

    Returns PIL images and int labels.  No transforms applied in __getitem__
    so label scans are fast.
    """

    def __init__(self, num_classes=100, samples_per_class=500, seed=42):
        super().__init__()
        rng = np.random.RandomState(seed)
        palette = rng.rand(num_classes, 3).astype(np.float32)
        textures = rng.randn(num_classes, 8, 8, 3).astype(np.float32)

        data = []
        labels = []
        for cls in range(num_classes):
            tex = textures[cls]
            tex_up = np.repeat(np.repeat(tex, 4, axis=0), 4, axis=1)
            for s in range(samples_per_class):
                shift = rng.randn(32, 32, 3).astype(np.float32) * 0.15
                img = np.clip(palette[cls] + tex_up * 0.3 + shift, 0, 1)
                data.append((img * 255).astype(np.uint8))
                labels.append(cls)
        self.data = np.stack(data)  # (N, 32, 32, 3) uint8
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = self.data[idx]
        return TF.to_pil_image(arr), int(self.labels[idx])


def load_cifar100():
    """Try real CIFAR-100 (already-downloaded); fall back to synthetic."""
    cifar_dir = os.path.join("data", "cifar-100-python")
    if os.path.isdir(cifar_dir) and any(
        fname in os.listdir(cifar_dir) for fname in ("train", "test", "meta")
    ):
        try:
            print("  Loading already-downloaded CIFAR-100 ...")
            full_train = torchvision.datasets.CIFAR100(
                root="./data", train=True, download=False
            )
            full_test = torchvision.datasets.CIFAR100(
                root="./data", train=False, download=False
            )
            print("  Real CIFAR-100 loaded successfully.")
            return full_train, full_test, False
        except Exception as e:
            print(f"  CIFAR-100 load failed: {e}")

    print("  Real CIFAR-100 data not found (or download would be required).")
    print("  Falling back to synthetic CIFAR-100 data.")
    sys.stdout.flush()
    syn_train = SyntheticCIFAR100(num_classes=100, samples_per_class=500, seed=SEED)
    syn_test = SyntheticCIFAR100(num_classes=100, samples_per_class=100, seed=SEED + 1)
    return syn_train, syn_test, True


# =====================================================================
#  Data Preparation
# =====================================================================
print("=" * 60)
print("Preparing data ...")
print("=" * 60)

full_train, full_test, using_synthetic = load_cifar100()

# ---- Pick 20 random classes -----------------------------------------
all_classes = list(range(100))
random.shuffle(all_classes)
selected_classes = sorted(all_classes[:NUM_CLASSES])
print(f"Selected class ids: {selected_classes}")

class_to_idx = {old: new for new, old in enumerate(selected_classes)}

# Scan raw labels (fast: numpy array for synthetic, PIL output for real CIFAR-100)
if using_synthetic:
    train_labels_arr = full_train.labels
    test_labels_arr = full_test.labels
    train_indices = []
    test_indices = []
    for cls in selected_classes:
        c_train = np.where(train_labels_arr == cls)[0].tolist()
        c_test = np.where(test_labels_arr == cls)[0].tolist()
        random.shuffle(c_train)
        random.shuffle(c_test)
        train_indices.extend(c_train[:NUM_TRAIN_PER_CLASS])
        test_indices.extend(c_test[:NUM_TEST_PER_CLASS])
else:
    train_indices = []
    test_indices = []
    for cls in selected_classes:
        c_train = [i for i, (_, lbl) in enumerate(full_train) if lbl == cls]
        c_test = [i for i, (_, lbl) in enumerate(full_test) if lbl == cls]
        random.shuffle(c_train)
        random.shuffle(c_test)
        train_indices.extend(c_train[:NUM_TRAIN_PER_CLASS])
        test_indices.extend(c_test[:NUM_TEST_PER_CLASS])

print(f"  Found {len(train_indices)} train indices, {len(test_indices)} test indices")


class SubsetWithTransform(Dataset):
    """Custom Subset that applies transforms + label remapping.

    Extends Dataset (not Subset) to work around PyTorch 2.6 DataLoader behavior
    where Subset.__getitem__ is bypassed when num_workers=0.
    """

    def __init__(self, dataset, indices, label_map, transform=None):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        else:
            x = TF.to_tensor(x)
        return x, self.label_map[int(y)]


# Build transforms
train_transform = T.Compose(
    [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)
test_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)

train_dataset = SubsetWithTransform(
    full_train, train_indices, class_to_idx, transform=train_transform
)
test_dataset = SubsetWithTransform(
    full_test, test_indices, class_to_idx, transform=test_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)

print(f"Train samples: {len(train_dataset)}  (20 per class x 20 classes)")
print(f"Test samples:  {len(test_dataset)}  (50 per class x 20 classes)")

# =====================================================================
#  Model Definitions
# =====================================================================


class LoRAConv2d(nn.Module):
    """Conv2d wrapper with low-rank adaptation."""

    def __init__(self, conv: nn.Conv2d, rank: int = 4, alpha: int = 8):
        super().__init__()
        self.conv = conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        out_ch, in_ch, k_h, k_w = conv.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(out_ch, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, in_ch * k_h * k_w))

        nn.init.kaiming_uniform_(self.lora_A)

        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_weight = (self.lora_A @ self.lora_B).view(self.conv.weight.shape)
        return self.conv(x) + (
            F.conv2d(
                x,
                delta_weight,
                bias=None,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )
            * self.scaling
        )


def _replace_conv_lora(model: nn.Module, rank: int = 4, alpha: int = 8):
    """Recursively replace every nn.Conv2d in *model* with LoRAConv2d."""
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(model, name, LoRAConv2d(child, rank, alpha))
        else:
            _replace_conv_lora(child, rank, alpha)


def make_model(pretrained: bool = False, use_lora: bool = False) -> nn.Module:
    """Build a ResNet-18 adapted for CIFAR-100 (32x32 images, 20 classes)."""
    weights = "IMAGENET1K_V1" if pretrained else None
    model = torchvision.models.resnet18(weights=weights)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    if use_lora:
        _replace_conv_lora(model)

    return model


def freeze_backbone(model: nn.Module):
    """Freeze all parameters except fc (linear-probe style)."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def freeze_except_lora_and_fc(model: nn.Module):
    """Only LoRA parameters and the classifier head remain trainable."""
    for name, p in model.named_parameters():
        p.requires_grad = ("lora" in name) or ("fc" in name)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
#  Training Utilities
# =====================================================================
def train_epoch(model, loader, optimizer, criterion) -> tuple:
    """One training epoch.  Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion) -> tuple:
    """Evaluation.  Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def extract_features(model, loader):
    """Extract penultimate-layer (after avgpool) features on test set."""
    model.eval()
    hook_outputs = []

    def hook_fn(module, inp, out):
        hook_outputs.append(out)

    handle = model.avgpool.register_forward_hook(hook_fn)

    labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        model(x)
        labels.append(y)
    handle.remove()

    features = torch.cat(hook_outputs, dim=0).squeeze().cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels


# =====================================================================
#  Experiment Runner
# =====================================================================
def run_experiment(name, pretrained, use_lora, freeze_fn, lr) -> dict:
    """Train one variant and return dict with history, trainable count, model."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    model = make_model(pretrained=pretrained, use_lora=use_lora).to(device)
    if freeze_fn is not None:
        freeze_fn(model)

    trainable = count_trainable(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / Total: {total:,}")

    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(opt_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(EPOCHS):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion)
        vl, va = eval_epoch(model, test_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_loss"].append(vl)
        history["test_acc"].append(va)

        print(
            f"  Epoch {epoch + 1:2d}/{EPOCHS}  "
            f"train loss {tl:.4f}  acc {ta:.4f}  |  "
            f"test loss {vl:.4f}  acc {va:.4f}  |  "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

    best_acc = max(history["test_acc"])
    print(f"  >> Best test acc: {best_acc:.4f}")

    return {
        "history": history,
        "trainable_params": trainable,
        "best_acc": best_acc,
        "model": model,
    }


# =====================================================================
#  Visualizations
# =====================================================================
def plot_accuracy_bar(all_results, output_dir):
    """Bar chart: best test accuracy for each strategy."""
    methods = list(all_results.keys())
    labels = ["Scratch", "Linear\nProbe", "Full\nFinetune", "LoRA"]
    best_accs = [all_results[m]["best_acc"] * 100 for m in methods]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, best_accs, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, acc in zip(bars, best_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Best Test Accuracy (%)", fontsize=12)
    data_source = "Synthetic" if using_synthetic else "CIFAR-100"
    ax.set_title(
        f"Transfer Learning on {data_source}\n(20 classes, 20 shots/class)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, max(best_accs) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_bar.png"), dpi=150)
    plt.close(fig)
    print("  Saved  accuracy_bar.png")


def plot_convergence(all_results, output_dir):
    """Convergence curves: test accuracy vs epoch for all strategies."""
    methods = list(all_results.keys())
    labels = ["Scratch", "Linear Probe", "Full Finetune", "LoRA"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(methods):
        accs = [a * 100 for a in all_results[name]["history"]["test_acc"]]
        ax.plot(range(1, EPOCHS + 1), accs, color=colors[i], linewidth=2.0, label=labels[i])

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Convergence Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence.png"), dpi=150)
    plt.close(fig)
    print("  Saved  convergence.png")


def plot_params_vs_accuracy(all_results, output_dir):
    """Scatter plot: trainable params (log) vs best accuracy."""
    methods = list(all_results.keys())
    labels_list = ["Scratch", "Linear Probe", "Full Finetune", "LoRA"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    params = [all_results[m]["trainable_params"] for m in methods]
    accs = [all_results[m]["best_acc"] * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(methods):
        ax.scatter(params[i], accs[i], s=140, c=colors[i], edgecolors="black", linewidths=0.5, zorder=5)
        ax.text(params[i] * 1.08, accs[i], labels_list[i], fontsize=10, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=12)
    ax.set_ylabel("Best Test Accuracy (%)", fontsize=12)
    ax.set_title("Parameter Efficiency", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for p, a, lab in zip(params, accs, labels_list):
        ax.text(p, a + 1.2, f"({p:,}, {a:.1f}%)", fontsize=8, ha="center", va="bottom", alpha=0.7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "params_vs_accuracy.png"), dpi=150)
    plt.close(fig)
    print("  Saved  params_vs_accuracy.png")


def plot_tsne(all_results, output_dir):
    """2x2 t-SNE panel: features of each method colored by ground-truth class."""
    methods = list(all_results.keys())
    labels_list = ["Scratch", "Linear Probe", "Full Finetune", "LoRA"]
    colors_method = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for idx, name in enumerate(methods):
        print(f"  Computing t-SNE for {labels_list[idx]} ...")
        model = all_results[name]["model"]
        feats, labs = extract_features(model, test_loader)

        # PCA pre-reduction for speed
        pca = PCA(n_components=min(50, len(feats) - 1), random_state=SEED)
        feats = pca.fit_transform(feats)

        perplexity = min(30, len(feats) - 1)
        tsne = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=perplexity,
            max_iter=500,
            init="random",
            learning_rate="auto",
        )
        embeds = tsne.fit_transform(feats)

        ax = axes_flat[idx]
        ax.scatter(embeds[:, 0], embeds[:, 1], c=labs, cmap="tab20", s=10, alpha=0.85)
        ax.set_title(labels_list[idx], fontsize=11, fontweight="bold", color=colors_method[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("t-SNE of Penultimate-Layer Features", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tsne.png"), dpi=150)
    plt.close(fig)
    print("  Saved  tsne.png")


# =====================================================================
#  Main
# =====================================================================
def main():
    # ---- Run all 4 strategies ----------------------------------------
    strategies = [
        # (name, pretrained, use_lora, freeze_fn,           lr)
        ("1-Scratch", False, False, None, 0.01),
        ("2-LinearProbe", True, False, freeze_backbone, 0.001),
        ("3-FullFinetune", True, False, None, 0.01),
        ("4-LoRA", True, True, freeze_except_lora_and_fc, 0.001),
    ]

    all_results = {}
    for name, pretrained, use_lora, freeze_fn, lr in strategies:
        result = run_experiment(name, pretrained, use_lora, freeze_fn, lr)
        all_results[name] = result

    # ---- Summary table -----------------------------------------------
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Method':<20} {'Trainable Params':<20} {'Best Acc':<10}")
    print(f"  {'-' * 50}")
    for name in all_results:
        short = name.split("-", 1)[1]
        r = all_results[name]
        print(f"  {short:<20} {r['trainable_params']:<20,} {r['best_acc'] * 100:.2f}%")

    # ---- Visualizations ----------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Generating visualizations ...")
    print(f"{'=' * 60}")

    plot_accuracy_bar(all_results, OUTPUT_DIR)
    plot_convergence(all_results, OUTPUT_DIR)
    plot_params_vs_accuracy(all_results, OUTPUT_DIR)
    plot_tsne(all_results, OUTPUT_DIR)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
