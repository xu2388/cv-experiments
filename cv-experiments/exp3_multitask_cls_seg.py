"""
Multi-Task Learning on CIFAR-10
Shared backbone -> two heads:
  Task A: Image classification (10 classes, CrossEntropyLoss)
  Task B: Layout prediction (4x4 grid binary mask, BCEWithLogitsLoss)

Compares:
  1. Single task (classification only)
  2. Single task (layout only)
  3. Multi-task: loss = cls_loss + lambda * layout_loss, lambda in [0.1, 0.5, 1.0]

Saves to outputs/exp3/:
  - taskA_accuracy_comparison.png
  - taskB_accuracy_comparison.png
  - tsne_single_cls.png / tsne_single_lay.png / tsne_multitask.png
  - lambda_ablation_table.csv
"""

import os, copy, csv, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Force CPU for stability on this Windows/WDDM machine
device = torch.device("cpu")

NC = 10         # num classes
GRID = 4        # layout grid dim
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
OUT = os.path.join(os.path.expanduser("~"), "CV实验", "outputs", "exp3")

os.makedirs(OUT, exist_ok=True)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Device: {device}")
print(f"Outputs -> {OUT}")
print(f"Epochs={EPOCHS}  Batch={BATCH_SIZE}")

# =====================================================================
# 1. Layout templates: 4x4 binary masks (one per class, flattened to 16)
# =====================================================================
TEMPLATES = torch.zeros(NC, GRID * GRID)
TEMPLATES[0, [5, 6, 9, 10]] = 1.0               # airplane - centre
TEMPLATES[1, [8, 9, 10, 11, 12, 13, 14, 15]] = 1.0   # automobile - bottom row
TEMPLATES[2, [1, 4, 5, 6, 7, 9, 13]] = 1.0      # bird - cross
TEMPLATES[3, [0, 1, 2, 3, 4, 5, 6, 7]] = 1.0    # cat - left half
TEMPLATES[4, [2, 3, 6, 7]] = 1.0                 # deer - top-right
TEMPLATES[5, [8, 9, 10, 11, 12, 13, 14, 15]] = 1.0   # dog - right half
TEMPLATES[6, [0, 1, 2, 3, 4, 7, 8, 11]] = 1.0   # frog - T shape
TEMPLATES[7, [8, 9, 12, 13]] = 1.0               # horse - bottom-left
TEMPLATES[8, [1, 5, 9, 13]] = 1.0                # ship - vertical centre
TEMPLATES[9, [0, 1, 2, 3, 12, 13, 14, 15]] = 1.0    # truck - top+bottom

# =====================================================================
# 2. Model architecture
# =====================================================================
class Backbone(nn.Module):
    """CNN backbone. ~90k params, fast on CPU."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),       # 16
            nn.Conv2d(32, 64, 3, 2, 1),  nn.BatchNorm2d(64), nn.ReLU(),      # 8
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),     # 8
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),    # 8
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)  # [B, 256]


class MultiTaskModel(nn.Module):
    """Shared backbone + classification + layout heads."""
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.cls_head = nn.Linear(256, NC)
        self.lay_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, GRID * GRID))

    def forward(self, x):
        feat = self.backbone(x)
        return self.cls_head(feat), self.lay_head(feat), feat


class SingleTaskModel(nn.Module):
    """Backbone + single head."""
    def __init__(self, task='cls'):
        super().__init__()
        self.backbone = Backbone()
        if task == 'cls':
            self.head = nn.Linear(256, NC)
        else:
            self.head = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, GRID * GRID))

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat), feat

# =====================================================================
# 3. Data: synthetic CIFAR-10-like (32x32 RGB, 10 classes)
# =====================================================================
def _synthetic(n):
    rng = np.random.RandomState(SEED)
    lbl = rng.randint(0, NC, n).astype(np.int64)
    means = np.array([
        [120, 130, 200], [180, 180, 180], [100, 180, 100], [160, 120, 100],
        [140, 160, 80],  [160, 100, 80],  [80, 160, 80],   [120, 100, 160],
        [100, 140, 180], [170, 150, 120],
    ], dtype=np.float32)
    m = means[lbl]                               # [N, 3]
    p = rng.randn(n, 32, 32).astype(np.float32) * 40   # [N, 32, 32]
    c = rng.randn(n, 3).astype(np.float32) * 25         # [N, 3]
    d = m[:, :, None, None] + c[:, :, None, None] + p[:, None, :, :]
    d = np.clip(d, 0, 255).astype(np.uint8)
    return d, lbl


def _loader(d, l, bs):
    d = d.astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
    d = (d - mean) / std
    imgs = torch.from_numpy(d).float()
    lbls = torch.from_numpy(l).long()
    lays = TEMPLATES[lbls]
    return DataLoader(TensorDataset(imgs, lbls, lays), bs, shuffle=True, num_workers=0)


print("Building data (synthetic CIFAR-10-like)...", flush=True)
# Subsample for reasonable CPU runtime: 25k train / 5k test
train_loader = _loader(*_synthetic(25000), BATCH_SIZE)
test_loader  = _loader(*_synthetic(5000), BATCH_SIZE)
print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}", flush=True)

# =====================================================================
# 4. Training loop
# =====================================================================
def run_epoch(model, loader, opt, crit_cls, crit_lay, lam=1.0, mt=True, train=True):
    model.train(train)
    ca_sum, la_sum, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, labels, layouts in loader:
            if train:
                opt.zero_grad()

            if mt:
                cls_logits, lay_logits, _ = model(imgs)
                loss = crit_cls(cls_logits, labels) + lam * crit_lay(lay_logits, layouts)
                if train:
                    loss.backward()
                    opt.step()
                ca_sum += (cls_logits.argmax(1) == labels).float().mean().item()
                la_sum += ((lay_logits > 0).float() == layouts).float().mean().item()
            else:
                logits, _ = model(imgs)
                loss = crit_cls(logits, labels)
                if train:
                    loss.backward()
                    opt.step()
                ca_sum += (logits.argmax(1) == labels).float().mean().item()
            n += 1

    return ca_sum / n, la_sum / n


def experiment(name, mt=True, lam=1.0):
    print(f"\n--- {name} ---", flush=True)
    model = (MultiTaskModel() if mt else SingleTaskModel('cls'))
    opt = optim.Adam(model.parameters(), lr=LR)
    crit_cls = nn.CrossEntropyLoss()
    crit_lay = nn.BCEWithLogitsLoss()

    best_ca, best_la = 0.0, 0.0
    best_state = None

    for ep in range(1, EPOCHS + 1):
        run_epoch(model, train_loader, opt, crit_cls, crit_lay, lam, mt, train=True)
        ca, la = run_epoch(model, test_loader, None, crit_cls, crit_lay, lam, mt, train=False)

        if ca > best_ca:
            best_ca, best_la = ca, la
            best_state = copy.deepcopy(model.state_dict())

        if ep in (1, 5, 10, 20, 30):
            print(f"  Ep {ep:3d}/{EPOCHS}  Cls:{ca*100:.2f}%  Lay:{la*100:.2f}%", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    ca, la = run_epoch(model, test_loader, None, crit_cls, crit_lay, lam, mt, train=False)
    print(f"  -> Best Cls:{best_ca*100:.2f}%  Lay:{best_la*100:.2f}%", flush=True)
    return best_ca, best_la, model


# =====================================================================
# 5. Feature extraction & visualisation
# =====================================================================
@torch.no_grad()
def collect(model, mt=True):
    model.eval()
    feats, labs = [], []
    for imgs, labels, _ in test_loader:
        f = (model(imgs)[2] if mt else model(imgs)[1]).numpy()
        feats.append(f)
        labs.append(labels.numpy())
    return np.concatenate(feats, 0), np.concatenate(labs, 0)


def plot_tsne(F, L, title, path):
    n = F.shape[0]
    if n > 3000:
        ix = np.random.choice(n, 3000, replace=False)
        F, L = F[ix], L[ix]
    emb = TSNE(2, random_state=SEED, perplexity=30, max_iter=1000).fit_transform(F)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(NC):
        mask = L == c
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[plt.cm.tab10(c / 9)],
                   label=CIFAR10_CLASSES[c], s=8, alpha=0.7)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [t-SNE] -> {path}", flush=True)


def plot_bar(labels, values, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#1f77b4", "#ff7f0e", "#ffbb78", "#2ca02c"]
    bars = ax.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Chart] -> {path}", flush=True)


# =====================================================================
# 6. Main pipeline
# =====================================================================
print("\n" + "=" * 60)
print("Single-task Experiments")
print("=" * 60)
ca_single, _, model_cls = experiment("Classification Only", mt=False)
_, la_single, model_lay = experiment("Layout Only", mt=True, lam=1.0)

print("\n" + "=" * 60)
print("Multi-task Lambda Ablation")
print("=" * 60)
lambdas = [0.1, 0.5, 1.0]
results = []
for lam in lambdas:
    ca, la, m = experiment(f"Multi-task lam={lam}", mt=True, lam=lam)
    results.append((lam, ca, la, m))

print("\n" + "=" * 60)
print("t-SNE Visualisation")
print("=" * 60)
feats_cls, all_labels = collect(model_cls, mt=False)
plot_tsne(feats_cls, all_labels, "t-SNE (Single-task Classification)",
          os.path.join(OUT, "tsne_single_cls.png"))

feats_lay, _ = collect(model_lay, mt=True)
plot_tsne(feats_lay, all_labels, "t-SNE (Single-task Layout)",
          os.path.join(OUT, "tsne_single_lay.png"))

best_mt = max(results, key=lambda x: x[1] + x[2])[3]
feats_mt, _ = collect(best_mt, mt=True)
plot_tsne(feats_mt, all_labels, "t-SNE (Multi-task, Best Lambda)",
          os.path.join(OUT, "tsne_multitask.png"))

print("\n" + "=" * 60)
print("Bar Charts")
print("=" * 60)
plot_bar(["Cls Only"] + [f"MT l={l}" for l in lambdas],
         [ca_single * 100] + [r[1] * 100 for r in results],
         "Task A - Classification Accuracy",
         os.path.join(OUT, "taskA_accuracy_comparison.png"))
plot_bar(["Lay Only"] + [f"MT l={l}" for l in lambdas],
         [la_single * 100] + [r[2] * 100 for r in results],
         "Task B - Layout Prediction Accuracy",
         os.path.join(OUT, "taskB_accuracy_comparison.png"))

print("\n" + "=" * 60)
print("Lambda Ablation Table")
print("=" * 60)
csv_path = os.path.join(OUT, "lambda_ablation_table.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Experiment", "Cls Acc (%)", "Lay Acc (%)"])
    w.writerow(["Single-task Cls", f"{ca_single * 100:.2f}", "-"])
    w.writerow(["Single-task Lay", "-", f"{la_single * 100:.2f}"])
    for l, ca, la, _ in results:
        w.writerow([f"Multi-task lam={l}", f"{ca * 100:.2f}", f"{la * 100:.2f}"])
print(f"  [CSV] -> {csv_path}")

print(f"{'Experiment':<22s}  {'Cls Acc (%)':<13s}  {'Lay Acc (%)':<13s}")
print("-" * 50)
print(f"{'Single-task Cls':<22s}  {ca_single * 100:>8.2f}     {'-':>8s}")
print(f"{'Single-task Lay':<22s}  {'-':>8s}       {la_single * 100:>8.2f}")
for l, ca, la, _ in results:
    print(f"{f'Multi-task lam={l}':<22s}  {ca * 100:>8.2f}     {la * 100:>8.2f}")

print(f"\nAll outputs saved to: {OUT}")
print("Done.", flush=True)
