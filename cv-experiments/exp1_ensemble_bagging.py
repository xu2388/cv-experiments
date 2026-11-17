"""
Ensemble Learning (Bagging) on CIFAR-10 with ResNet-18.

Trains 5 independent ResNet-18 models with bootstrap samples,
then compares hard voting, soft voting, and single-model accuracy.
"""
import argparse
import os
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, models, transforms
# Monkey-patch integrity check for CIFAR-10
datasets.CIFAR10._check_integrity = lambda self: True

warnings.filterwarnings("ignore")

# ──────── Performance ──────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
#  Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
#  Data  (CIFAR-10)  — note ResNet-18 from torchvision expects 224×224
#  We keep 32×32 and adjust the first conv layer in the model factory.
# ---------------------------------------------------------------------------

def get_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])


def _patch_resnet18(num_classes: int = 10) -> nn.Module:
    """Build ResNet-18 and patch first conv for 32×32 CIFAR-10."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    # CIFAR images are 32×32 → replace 7×7 stride-2 conv with 3×3 stride-1,
    # and remove the initial max-pool.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _bootstrap_sampler(dataset_size: int, seed: int):
    """Return a SubsetRandomSampler that draws a bootstrap sample."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, dataset_size, size=dataset_size).tolist()
    return SubsetRandomSampler(indices)


def get_cifar10_loaders(
    batch_size: int,
    num_workers: int,
    seed: int,
    bootstrap: bool = True,
):
    """Return (train_loader, val_loader)."""
    root = r"D:\大三上的资料\假期学习记录"  # contains cifar-10-batches-py/
    train_dataset = datasets.CIFAR10(root, train=True,  download=False,
                                     transform=get_transform(train=True))
    val_dataset   = datasets.CIFAR10(root, train=False, download=False,
                                     transform=get_transform(train=False))

    if bootstrap:
        sampler = _bootstrap_sampler(len(train_dataset), seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
#  Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total   += y.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += logits.argmax(1).eq(y).sum().item()
        total   += y.size(0)
    return correct / total


@torch.no_grad()
def collect_logits_and_targets(models, loader, device):
    """Return (all_logits, all_targets) where all_logits shape (N, K, C)."""
    logits_list = []
    targets_list = []
    for x, y in loader:
        x = x.to(device)
        batch_logits = torch.stack([m(x) for m in models], dim=1)  # (B, K, C)
        logits_list.append(batch_logits.cpu())
        targets_list.append(y)
    return torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0)


@torch.no_grad()
def collect_features(models, loader, device):
    """Extract penultimate-layer features (after global avg pool, before FC)."""
    features_list = []
    targets_list = []

    def hook_fn(m, inp, out):
        features_list.append(out.detach().cpu())

    handles = []
    for model in models:
        handle = model.avgpool.register_forward_hook(hook_fn)
        handles.append(handle)

    all_y = []
    for x, y in loader:
        x = x.to(device)
        _ = [m(x) for m in models]
        all_y.append(y)

    for h in handles:
        h.remove()

    # features_list: M*N tensors, each shape (batch, 512, 1, 1)
    # We stack per-model, then average across models.
    feats = torch.cat(features_list, dim=0)  # (M*N, 512, 1, 1)
    feats = feats.view(len(models), -1, 512).mean(dim=0)  # (N, 512)
    targets = torch.cat(all_y, dim=0)
    return feats.numpy(), targets.numpy()


# ---------------------------------------------------------------------------
#  Ensemble predictions
# ---------------------------------------------------------------------------

def hard_voting(logits: torch.Tensor) -> torch.Tensor:
    """logits: (N, K, C) → predicted labels (N,) from majority vote."""
    preds = logits.argmax(dim=2)  # (N, K)
    majority, _ = preds.mode(dim=1)
    return majority


def soft_voting(logits: torch.Tensor) -> torch.Tensor:
    """logits: (N, K, C) → predicted labels (N,) from averaged softmax."""
    probs = F.softmax(logits, dim=2).mean(dim=1)  # (N, C)
    return probs.argmax(dim=1)


def accuracy_from_logits(logits, targets, strategy="soft"):
    if strategy == "hard":
        preds = hard_voting(logits)
    else:
        preds = soft_voting(logits)
    return preds.eq(targets).float().mean().item()


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_results(results, incremental_acc, tsne_data, save_dir):
    """Generate 3-panel figure."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ----- 1) Bar chart ----------------------------------------------------
    strategies = list(results.keys())
    accuracies = [results[k] * 100 for k in strategies]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    bars = axes[0].bar(strategies, accuracies, color=colors[:len(strategies)],
                       edgecolor="gray", linewidth=0.6)
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Ensemble Strategy Comparison", fontweight="bold")
    for bar, val in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")
    axes[0].set_ylim(0, max(accuracies) * 1.15)

    # ----- 2) Model count vs accuracy curve --------------------------------
    model_counts = list(range(1, len(incremental_acc) + 1))
    vals_inc = [v * 100 for v in incremental_acc]
    axes[1].plot(model_counts, vals_inc, marker="o", color="#e74c3c",
                 linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Models in Ensemble")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("Accuracy vs Ensemble Size", fontweight="bold")
    axes[1].set_xticks(model_counts)
    for i, (mc, va) in enumerate(zip(model_counts, vals_inc)):
        axes[1].annotate(f"{va:.2f}%", (mc, va),
                         textcoords="offset points", xytext=(0, 12),
                         ha="center", fontsize=9)

    # ----- 3) t-SNE of ensemble features -----------------------------------
    feats, targets = tsne_data
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(feats.shape[0]-1, 3)),
                init="pca", learning_rate="auto")
    emb = tsne.fit_transform(feats)
    scatter = axes[2].scatter(emb[:, 0], emb[:, 1], c=targets, cmap="tab10",
                              s=8, alpha=0.7)
    axes[2].set_title("t-SNE of Ensemble Features", fontweight="bold")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    cbar = fig.colorbar(scatter, ax=axes[2], ticks=range(10))
    cbar.set_label("Class")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "ensemble_results.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Figure saved to {save_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--lr",           type=float, default=0.1)
    parser.add_argument("--momentum",     type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-models",   type=int, default=5)
    parser.add_argument("--seeds",        type=int, nargs="+",
                        default=[42, 1337, 2021, 777, 12345])
    parser.add_argument("--output-dir",   type=str, default="outputs/exp1")
    parser.add_argument("--num-workers",  type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if len(args.seeds) < args.num_models:
        raise ValueError("Provide at least as many seeds as num_models")

    # -----------------------------------------------------------------------
    #  Train individual models
    # -----------------------------------------------------------------------
    models_list = []
    val_loaders = []           # keep a loader per model (same test set)

    for i in range(args.num_models):
        seed = args.seeds[i]
        set_seed(seed)
        print(f"\n{'='*60}", flush=True)
        print(f"  Training model {i+1}/{args.num_models}  (seed={seed})", flush=True)
        print(f"{'='*60}", flush=True)

        model = _patch_resnet18(num_classes=10).to(device)
        train_loader, val_loader = get_cifar10_loaders(
            args.batch_size, args.num_workers, seed, bootstrap=True,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_state = None
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_acc = evaluate(model, val_loader, device)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = deepcopy(model.state_dict())

            print(f"    Epoch {epoch:2d}/{args.epochs}  "
                  f"Train Loss {train_loss:.4f}  "
                  f"Train Acc {train_acc:.4f}  "
                  f"Val Acc {val_acc:.4f}  "
                  f"LR {scheduler.get_last_lr()[0]:.5f}", flush=True)

        # restore best checkpoint
        model.load_state_dict(best_state)
        models_list.append(model)
        val_loaders.append(val_loader)
        print(f"  >> Best val acc: {best_acc * 100:.2f}%")

    # -----------------------------------------------------------------------
    #  Single-model accuracies
    # -----------------------------------------------------------------------
    single_accs = []
    for i, model in enumerate(models_list):
        acc = evaluate(model, val_loaders[0], device)
        single_accs.append(acc)
        print(f"  Model {i+1} test accuracy: {acc * 100:.2f}%")

    best_single = max(single_accs)
    print(f"  >> Best single model accuracy: {best_single * 100:.2f}%")

    # -----------------------------------------------------------------------
    #  Ensemble evaluation (collect all logits once)
    # -----------------------------------------------------------------------
    all_logits, all_targets = collect_logits_and_targets(
        models_list, val_loaders[0], device
    )
    # all_logits: (N, K, C)

    hard_acc = accuracy_from_logits(all_logits, all_targets, "hard")
    soft_acc = accuracy_from_logits(all_logits, all_targets, "soft")
    print(f"  Hard voting accuracy: {hard_acc * 100:.2f}%")
    print(f"  Soft voting accuracy: {soft_acc * 100:.2f}%")

    # -----------------------------------------------------------------------
    #  Incremental ensemble accuracy (models added 1 → K)
    # -----------------------------------------------------------------------
    incremental = []
    for k in range(1, args.num_models + 1):
        sub_logits = all_logits[:, :k, :]
        acc = accuracy_from_logits(sub_logits, all_targets, "soft")
        incremental.append(acc)
        print(f"  Ensemble size {k}: accuracy = {acc * 100:.2f}%")

    # -----------------------------------------------------------------------
    #  t-SNE features (average of ensemble features)
    # -----------------------------------------------------------------------
    print("\n[*] Extracting features for t-SNE ...")
    tsne_feats, tsne_targets = collect_features(
        models_list, val_loaders[0], device
    )
    # Subsample if too many points
    n_total = tsne_feats.shape[0]
    if n_total > 3000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, 3000, replace=False)
        tsne_feats = tsne_feats[idx]
        tsne_targets = tsne_targets[idx]

    # -----------------------------------------------------------------------
    #  Plot
    # -----------------------------------------------------------------------
    results = {
        "Best Single":     best_single,
        "Hard Voting":     hard_acc,
        "Soft Voting":     soft_acc,
    }
    plot_results(results, incremental, (tsne_feats, tsne_targets),
                 args.output_dir)

    # -----------------------------------------------------------------------
    #  Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Best single model accuracy:        {best_single * 100:.2f}%")
    print(f"  Hard voting ensemble accuracy:     {hard_acc * 100:.2f}%")
    print(f"  Soft voting ensemble accuracy:     {soft_acc * 100:.2f}%")
    improve_hard = (hard_acc - best_single) * 100
    improve_soft = (soft_acc - best_single) * 100
    print(f"  Improvement over single (hard):    {improve_hard:+.2f}%")
    print(f"  Improvement over single (soft):    {improve_soft:+.2f}%")
    for k in range(1, args.num_models + 1):
        print(f"  Ensemble of {k} models:               {incremental[k-1] * 100:.2f}%")
    print(f"\n  Figure saved to {os.path.join(args.output_dir, 'ensemble_results.png')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
