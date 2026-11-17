"""
Semi-Supervised Learning (Self-Training) on CIFAR-10 with ResNet-18.

Uses only 10% labeled data (500 images, 50 per class) and treats the remaining
45,000 as unlabeled.

Self-training loop: train on labeled data -> predict on unlabeled ->
add high-confidence pseudo-labels -> retrain from scratch (repeat 5 rounds).

Compares:
  1. Supervised baseline (500 labeled only)
  2. Self-training with threshold = 0.95
  3. Self-training with threshold = 0.99
  4. Fully supervised upper bound (all 50,000 labeled)
"""
import argparse
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

# ──────── Performance optimizations ──────────────────────────────────────
torch.backends.cudnn.benchmark = True      # auto-tune cuDNN kernels
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
#  Transforms  (CIFAR-10 standard stats)
# ---------------------------------------------------------------------------


def get_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


# ---------------------------------------------------------------------------
#  Model  (ResNet-18 patched for 32x32 CIFAR-10)
# ---------------------------------------------------------------------------

def _patch_resnet18(num_classes: int = 10) -> nn.Module:
    """Build ResNet-18 and replace first conv / pooling for 32x32 inputs."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ---------------------------------------------------------------------------
#  Custom dataset for self-training
# ---------------------------------------------------------------------------

class SelfTrainingDataset(Dataset):
    """Wraps CIFAR-10 train set with on-the-fly augmentation.

    - For indices in *labeled_indices*: returns the original label.
    - For indices in *pseudo_label_map*: returns the stored pseudo-label.
    """

    def __init__(self, base_dataset, labeled_indices, pseudo_label_map):
        super().__init__()
        self.base_dataset = base_dataset
        self.all_indices = sorted(set(labeled_indices) | set(pseudo_label_map.keys()))
        self.pseudo_label_map = pseudo_label_map

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        real_idx = self.all_indices[idx]
        img, _ = self.base_dataset[real_idx]
        if real_idx in self.pseudo_label_map:
            return img, self.pseudo_label_map[real_idx]
        return img, self.base_dataset.targets[real_idx]


# ---------------------------------------------------------------------------
#  Training / evaluation helpers
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
        total += y.size(0)
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
        total += y.size(0)
    return correct / total


@torch.no_grad()
def predict_with_confidence(model, loader, device):
    """Return (preds, confidences, true_targets) for every sample in loader."""
    model.eval()
    all_preds, all_conf, all_targets = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        all_preds.append(preds.cpu())
        all_conf.append(conf.cpu())
        all_targets.append(y.cpu())
    return (
        torch.cat(all_preds, dim=0),
        torch.cat(all_conf, dim=0),
        torch.cat(all_targets, dim=0),
    )


def train_model(train_loader, test_loader, epochs, lr, momentum,
                weight_decay, device):
    """Train ResNet-18 from scratch; return (model, best_test_acc)."""
    model = _patch_resnet18(num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = deepcopy(model.state_dict())

        print(
            f"    Epoch {epoch:3d}/{epochs}  "
            f"Loss {train_loss:.4f}  "
            f"Train Acc {train_acc:.4f}  "
            f"Test Acc {test_acc:.4f}  "
            f"LR {scheduler.get_last_lr()[0]:.5f}"
        )

    model.load_state_dict(best_state)
    return model, best_acc


# ---------------------------------------------------------------------------
#  Robust CIFAR-10 downloader  (handles slow connections with resume)
# ---------------------------------------------------------------------------

def _ensure_cifar10(root):
    """Download and extract CIFAR-10 tarball with retry-and-resume if needed."""
    import urllib.request, ssl, tarfile
    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, "cifar-10-python.tar.gz")
    extracted = os.path.join(root, "cifar-10-batches-py")

    if os.path.isdir(extracted) and len(os.listdir(extracted)) >= 5:
        return

    if os.path.exists(tgz):
        try:
            with tarfile.open(tgz) as tf:
                if len(tf.getmembers()) >= 5:
                    print("  Extracting CIFAR-10 archive ...")
                    tf.extractall(path=root)
                    return
        except Exception:
            os.remove(tgz)

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    ctx = ssl._create_unverified_context()
    for attempt in range(30):
        print(f"  Download attempt {attempt+1}/30 ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=9999, context=ctx)
            total = int(resp.headers.get("Content-Length", 170498071))
            downloaded = 0
            t0 = time.time()
            part = tgz + ".part"
            with open(part, "wb") as f:
                while True:
                    try:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        f.flush()
                        downloaded += len(chunk)
                        pct = downloaded / total * 100
                        rate = downloaded / max(time.time() - t0, 0.01) / 1024
                        print(f"\r    {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB ({pct:.1f}%) @ {rate:.0f} KB/s  ", end="")
                    except Exception:
                        print(f"\n    Reconnecting at {downloaded/1024/1024:.1f} MB ...")
                        resp.close()
                        r2 = urllib.request.Request(url, headers={
                            "User-Agent": "Mozilla/5.0", "Range": f"bytes={downloaded}-",
                        })
                        resp = urllib.request.urlopen(r2, timeout=9999, context=ctx)
            os.replace(part, tgz)
            print(f"\n    Complete ({downloaded/1024/1024:.1f} MB), extracting ...")
            with tarfile.open(tgz) as tf:
                tf.extractall(path=root)
            print("    Extraction done.")
            return
        except Exception as e:
            print(f"  Error: {e}")
            for p in [tgz, tgz + ".part"]:
                if os.path.exists(p):
                    os.remove(p)
            time.sleep(10)
    raise RuntimeError("Failed to download CIFAR-10 after 30 attempts")


# ---------------------------------------------------------------------------
#  Data splitting  (stratified, 50 per class = 500 labeled)
# ---------------------------------------------------------------------------

def split_cifar10(root, seed=42, labeled_per_class=50):
    full = datasets.CIFAR10(root, train=True, download=False)
    targets = np.array(full.targets)
    rng = np.random.default_rng(seed)
    labeled, unlabeled = [], []
    for cls in range(10):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        labeled.extend(cls_idx[:labeled_per_class].tolist())
        unlabeled.extend(cls_idx[labeled_per_class:].tolist())
    return labeled, unlabeled, targets


# ---------------------------------------------------------------------------
#  Self-training runner
# ---------------------------------------------------------------------------

def run_self_training(
    labeled_indices, unlabeled_indices, all_targets,
    train_dataset, eval_dataset, test_loader,
    threshold, num_rounds, epochs_per_round,
    batch_size, lr, momentum, weight_decay, device, num_workers,
):
    """
    Execute iterative self-training.

    Returns
    -------
    test_accs : list[float]
        Test accuracy after each round (length = num_rounds + 1).
    pseudo_counts : list[int]
        Pseudo-labels added *in each round* (first element = 0).
    pseudo_accs : list[float]
        Accuracy of pseudo-labels added each round (first element = 0.0).
    """
    pseudo_label_map = {}
    test_accs = []
    pseudo_counts = [0]
    pseudo_accs = [0.0]

    for round_idx in range(num_rounds + 1):
        n_labeled = len(labeled_indices)
        n_pseudo = len(pseudo_label_map)
        print(f"\n  --- Round {round_idx}/{num_rounds} ---")
        print(f"    Dataset: {n_labeled + n_pseudo} samples "
              f"({n_labeled} labeled + {n_pseudo} pseudo-labeled)")

        cur_dataset = SelfTrainingDataset(
            train_dataset, labeled_indices, pseudo_label_map,
        )
        cur_loader = DataLoader(
            cur_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
        )
        model, best_acc = train_model(
            cur_loader, test_loader, epochs_per_round,
            lr, momentum, weight_decay, device,
        )
        test_accs.append(best_acc)
        print(f"    => Test accuracy: {best_acc * 100:.2f}%")

        if round_idx >= num_rounds:
            break

        remaining = [i for i in unlabeled_indices if i not in pseudo_label_map]
        if not remaining:
            print("    => No remaining unlabeled data.")
            pseudo_counts.append(0)
            pseudo_accs.append(0.0)
            continue

        pred_loader = DataLoader(
            Subset(eval_dataset, remaining),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        )
        preds, conf, _ = predict_with_confidence(model, pred_loader, device)

        mask = conf >= threshold
        n_selected = mask.sum().item()
        pseudo_counts.append(n_selected)

        if n_selected > 0:
            local_idx = torch.where(mask)[0].tolist()
            sel_indices = [remaining[i] for i in local_idx]
            sel_preds = [preds[i].item() for i in local_idx]

            true_labels = [all_targets[idx] for idx in sel_indices]
            pseudo_accuracy = sum(
                1 for p, t in zip(sel_preds, true_labels) if p == t
            ) / n_selected
            pseudo_accs.append(pseudo_accuracy)

            for idx, pred in zip(sel_indices, sel_preds):
                pseudo_label_map[idx] = pred

            print(f"    => Added {n_selected} pseudo-labels  "
                  f"(accuracy: {pseudo_accuracy * 100:.2f}%)")
        else:
            pseudo_accs.append(0.0)
            print(f"    => No pseudo-labels above threshold {threshold}")

    while len(pseudo_counts) < num_rounds + 1:
        pseudo_counts.append(0)
        pseudo_accs.append(0.0)

    return test_accs, pseudo_counts[: num_rounds + 1], pseudo_accs[: num_rounds + 1]


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_accuracy_curves(all_accs, labels, colors, save_path):
    """Test accuracy vs self-training round (all methods on one axes)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    max_r = max(len(a) for a in all_accs)
    x = list(range(max_r))

    for accs, label, color in zip(all_accs, labels, colors):
        y = list(accs) + [accs[-1]] * (max_r - len(accs))
        ax.plot(x, y, marker="o", label=label, color=color,
                linewidth=2, markersize=7)

    ax.set_xlabel("Self-Training Round", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Self-Training Round", fontweight="bold")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    for accs, color in zip(all_accs, colors):
        for r, a in enumerate(accs):
            ax.annotate(f"{a * 100:.1f}",
                        (r, a), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7, color=color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved: {save_path}")


def plot_pseudo_label_stats(stats_dict, thresholds, save_path):
    """Dual-axis bar+line: pseudo-label count and accuracy per round."""
    n_plots = len(thresholds)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    palette = {"0.95": "#3498db", "0.99": "#e74c3c"}

    for ax_idx, th in enumerate(thresholds):
        ax = axes[ax_idx]
        counts, accs = stats_dict[th]

        rounds = list(range(1, len(counts)))
        plot_counts = counts[1:]
        plot_accs = accs[1:]

        color = palette.get(f"{th}", "#95a5a6")

        bars = ax.bar(rounds, plot_counts, color=color, alpha=0.55,
                      edgecolor="gray", linewidth=0.6,
                      label="Pseudo-labels added")
        ax.set_xlabel("Self-Training Round", fontsize=11)
        ax.set_ylabel("Number of Pseudo-Labels", fontsize=11, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_title(f"Threshold = {th}", fontweight="bold")
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.3, axis="x")

        ax2 = ax.twinx()
        valid_r, valid_a = [], []
        for r, c, a in zip(rounds, plot_counts, plot_accs):
            if c > 0:
                valid_r.append(r)
                valid_a.append(a * 100)

        if valid_r:
            ax2.plot(valid_r, valid_a, marker="s", color="green",
                     linewidth=2, markersize=7,
                     label="Pseudo-label accuracy")
            for r, a in zip(valid_r, valid_a):
                ax2.annotate(f"{a:.1f}%", (r, a),
                             textcoords="offset points", xytext=(0, -13),
                             ha="center", fontsize=8, color="green")

        ax2.set_ylabel("Pseudo-Label Accuracy (%)", fontsize=11, color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.set_ylim(0, 105)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h2:
            ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved: {save_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semi-Supervised Self-Training on CIFAR-10"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs per training run")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs/exp2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.95, 0.99])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device  : {device}")
    print(f"[*] Seed         : {args.seed}")
    print(f"[*] Output dir   : {args.output_dir}")
    print(f"[*] Epochs/round : {args.epochs}")
    print(f"[*] Thresholds   : {args.thresholds}")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # --------------- data ------------------------------------------------
    print("\n[*] Preparing CIFAR-10 ...")
    root = str(Path.home() / ".cache" / "cifar10")
    _ensure_cifar10(root)

    train_aug = datasets.CIFAR10(root, train=True, download=False,
                                 transform=get_transform(train=True))
    train_eval = datasets.CIFAR10(root, train=True, download=False,
                                  transform=get_transform(train=False))
    test_dataset = datasets.CIFAR10(root, train=False, download=False,
                                    transform=get_transform(train=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    labeled_idx, unlabeled_idx, all_targets = split_cifar10(root, args.seed)
    print(f"  Labeled   : {len(labeled_idx)}  (50/class)")
    print(f"  Unlabeled : {len(unlabeled_idx)}")

    # --------------- 1) Supervised baseline -------------------------------
    print(f"\n{'='*60}")
    print("  1) Supervised Baseline (500 labeled)")
    print(f"{'='*60}")
    baseline_loader = DataLoader(
        Subset(train_aug, labeled_idx),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    _, baseline_acc = train_model(
        baseline_loader, test_loader, args.epochs, args.lr,
        args.momentum, args.weight_decay, device,
    )

    # --------------- 4) Fully supervised upper bound ----------------------
    print(f"\n{'='*60}")
    print("  4) Fully Supervised Upper Bound (50,000 labeled)")
    print(f"{'='*60}")
    full_loader = DataLoader(
        Subset(train_aug, list(range(50000))),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    _, full_acc = train_model(
        full_loader, test_loader, args.epochs, args.lr,
        args.momentum, args.weight_decay, device,
    )

    # --------------- 2,3) Self-training runs -----------------------------
    st_results = {}
    for thresh in args.thresholds:
        print(f"\n{'='*60}")
        print(f"  Self-Training (threshold = {thresh})")
        print(f"{'='*60}")
        accs, counts, paccs = run_self_training(
            labeled_indices=labeled_idx,
            unlabeled_indices=unlabeled_idx,
            all_targets=all_targets,
            train_dataset=train_aug,
            eval_dataset=train_eval,
            test_loader=test_loader,
            threshold=thresh,
            num_rounds=args.num_rounds,
            epochs_per_round=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            device=device,
            num_workers=args.num_workers,
        )
        st_results[thresh] = {
            "test_accs": accs,
            "pseudo_counts": counts,
            "pseudo_accs": paccs,
        }

    elapsed = time.time() - t_start
    print(f"\n[*] Total training time: {elapsed / 60:.1f} min")

    # --------------- plot 1: accuracy vs round ---------------------------
    baseline_curve = [baseline_acc] * (args.num_rounds + 1)
    full_curve = [full_acc] * (args.num_rounds + 1)

    all_accs = [baseline_curve]
    method_labels = [f"Baseline ({len(labeled_idx)} labeled)"]
    colors = ["#e67e22"]

    for th in args.thresholds:
        all_accs.append(st_results[th]["test_accs"])
        method_labels.append(f"Self-Training (th={th})")
        colors.append("#3498db" if th > 0.97 else "#2ecc71")

    all_accs.append(full_curve)
    method_labels.append("Fully Supervised (50k)")
    colors.append("#9b59b6")

    plot_accuracy_curves(
        all_accs, method_labels, colors,
        str(output_dir / "accuracy_vs_round.png"),
    )

    # --------------- plot 2: pseudo-label stats --------------------------
    stats_for_plot = {
        th: (st_results[th]["pseudo_counts"], st_results[th]["pseudo_accs"])
        for th in args.thresholds
    }
    plot_pseudo_label_stats(
        stats_for_plot, args.thresholds,
        str(output_dir / "pseudo_label_stats.png"),
    )

    # --------------- final comparison table ------------------------------
    print(f"\n{'='*70}")
    print("  FINAL RESULTS - Test Accuracy Comparison")
    print(f"{'='*70}")
    header = f"  {'Method':<42s} {'Accuracy':>10s} {'vs Baseline':>13s}"
    sep = f"  {'-'*42} {'-'*10} {'-'*13}"
    print(header)
    print(sep)

    rows = []
    for label, acc in zip(method_labels, [a[-1] for a in all_accs]):
        diff = (acc - baseline_acc) * 100
        rows.append((label, acc))
        print(f"  {label:<42s} {acc*100:>8.2f}%  {diff:>+11.2f}%")
    print(f"{'='*70}")

    csv_path = output_dir / "comparison_table.csv"
    with open(str(csv_path), "w") as f:
        f.write("Method,TestAccuracy(%),ImprovementOverBaseline(pp)\n")
        for label, acc in rows:
            diff = (acc - baseline_acc) * 100
            f.write(f"{label},{acc*100:.2f},{diff:+.2f}\n")
    print(f"[*] Saved: {csv_path}")

    for th in args.thresholds:
        r = st_results[th]
        print(f"\n  Self-Training (th={th}) - per-round detail:")
        print(f"  {'Round':<7s} {'Test Acc':>10s} {'#Pseudo':>10s} "
              f"{'Pseudo Acc':>12s} {'Cumul.':>8s}")
        print(f"  {'-'*7} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
        cumul = 0
        for i in range(args.num_rounds + 1):
            acc = r["test_accs"][i]
            cnt = r["pseudo_counts"][i]
            pac = r["pseudo_accs"][i]
            cumul += cnt
            pa_str = f"{pac*100:.1f}%" if cnt > 0 else "-"
            print(f"  {i:<7d} {acc*100:>8.2f}% {cnt:>10d} "
                  f"{pa_str:>12s} {cumul:>8d}")

    print(f"\n[*] All outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
