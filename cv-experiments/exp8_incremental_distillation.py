#!/usr/bin/env python3
"""
Experiment 8: Incremental Semantic Segmentation with Knowledge Distillation

Two-task incremental learning on CIFAR-10 with synthetic segmentation masks.
  - Task A: classes 0-4 (airplane, automobile, bird, cat, deer)
  - Task B: classes 5-9 (dog, frog, horse, ship, truck)

For each class, a fixed geometric shape at image center is used as the
binary segmentation ground truth.

Model: Simple 3-level U-Net (1-channel binary segmentation output).

Methods compared:
  1. Fine-tune: train Task A, then fine-tune on Task B
  2. LwF: add distillation loss (KL divergence with temperature) on Task A data
  3. EWC: Fisher regularization on Task A parameters
  4. Joint: train A+B together (upper bound)

Outputs saved to outputs/exp8/:
  - per_class_iou.csv          -- Per-class IoU comparison table
  - forgetting_rate.png        -- Forgetting rate bar chart
  - seg_visualization.png      -- Segmentation visualization comparison
  - iou_vs_epochs.png          -- IoU vs training progress curve
  - all_results.json           -- Full numeric results
  - shape_templates.png        -- The 10 shape templates used
"""

import os
import sys
import copy
import json
import warnings
import math
import csv as csv_module

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 12})

# ═════════════════════════════════════════════════════════════════════════
#  0.  Setup & Constants
# ═════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "exp8")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Script starting...")

# Task definitions
TASK_A_CLASSES = list(range(5))      # 0,1,2,3,4
TASK_B_CLASSES = list(range(5, 10))  # 5,6,7,8,9
ALL_CLASSES = list(range(10))

CIFAR10_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Training hyper-parameters
TRAIN_EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
TEMPERATURE = 2.0
LAMBDA_DISTILL = 1.0   # weight for distillation loss in LwF
EWC_LAMBDA = 1000.0    # weight for EWC regularization
NUM_WORKERS = 0        # 0 for Windows safety
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Outputs dir: {OUTPUT_DIR}")

# Synthetic dataset size
NUM_TRAIN_PER_CLASS = 5000
NUM_TEST_PER_CLASS = 1000
FISHER_SUBSAMPLE = 1000  # samples used for Fisher computation
IMG_SIZE = 32

# ─────────────────────────── Reproducibility ───────────────────────────

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


# ═════════════════════════════════════════════════════════════════════════
#  1.  Synthetic Dataset: Images + Segmentation Masks
# ═════════════════════════════════════════════════════════════════════════

def _points_in_triangle(pts, v1, v2, v3):
    """Barycentric test: which points in (H,W,2) grid lie inside triangle."""
    def _cross(p, q, r):
        return ((q[0] - p[0]) * (r[..., 1] - p[1]) -
                (q[1] - p[1]) * (r[..., 0] - p[0]))
    # Sample at pixel centres (+0.5 offset) for robust integer-vertex triangles
    pts_c = pts.astype(np.float64) + 0.5
    d1 = _cross(v1, v2, pts_c)
    d2 = _cross(v2, v3, pts_c)
    d3 = _cross(v3, v1, pts_c)
    # Support both CW and CCW winding by checking either all >= 0 or all <= 0
    return ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))


def _points_in_convex_polygon(pts, vertices):
    """Test which points in a (H,W,2) grid lie inside a convex polygon."""
    n = len(vertices)
    inside = np.ones(pts.shape[:2], dtype=bool)
    pts_c = pts.astype(np.float64) + 0.5  # sample at pixel centres
    # First pass: check if the polygon is CW or CCW
    # Compute signed area to determine winding
    area = 0.0
    for i in range(n):
        p = vertices[i]
        q = vertices[(i + 1) % n]
        area += (q[0] - p[0]) * (q[1] + p[1])
    winding_is_ccw = area < 0  # positive area = CCW

    for i in range(n):
        p = vertices[i]
        q = vertices[(i + 1) % n]
        cross = ((q[0] - p[0]) * (pts_c[..., 1] - p[1]) -
                 (q[1] - p[1]) * (pts_c[..., 0] - p[0]))
        if winding_is_ccw:
            inside = inside & (cross >= 0)
        else:
            inside = inside & (cross <= 0)
    return inside


def _generate_shape_masks(img_size=IMG_SIZE):
    """Precompute binary masks for all 10 classes.

    Each class has a distinct geometric shape at the image centre.
    Returns: numpy array of shape (10, H, W)
    """
    c = img_size // 2
    r = img_size // 4
    hw = (r * 3) // 4
    rr = max(r - 2, 2)
    h = w = img_size

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([xs, ys], axis=-1)  # (H, W, 2)

    masks = []

    # 0: Circle (airplane)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[(xs - c) ** 2 + (ys - c) ** 2 <= r ** 2] = 1.0
    masks.append(mask)

    # 1: Square (automobile)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[c - r:c + r, c - r:c + r] = 1.0
    masks.append(mask)

    # 2: Triangle pointing up (bird)
    v1 = np.array([c, c - r])
    v2 = np.array([c - r, c + r])
    v3 = np.array([c + r, c + r])
    mask = np.zeros((h, w), dtype=np.float32)
    mask[_points_in_triangle(pts, v1, v2, v3)] = 1.0
    masks.append(mask)

    # 3: Diamond (cat)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[np.abs(xs - c) + np.abs(ys - c) <= r] = 1.0
    masks.append(mask)

    # 4: Plus / cross (deer)
    mask = np.zeros((h, w), dtype=np.float32)
    thickness = max(r // 2, 2)
    mask[c - r:c + r, c - thickness:c + thickness] = 1.0
    mask[c - thickness:c + thickness, c - r:c + r] = 1.0
    masks.append(mask)

    # 5: Triangle pointing down (dog)
    v1 = np.array([c - r, c - r])
    v2 = np.array([c + r, c - r])
    v3 = np.array([c, c + r])
    mask = np.zeros((h, w), dtype=np.float32)
    mask[_points_in_triangle(pts, v1, v2, v3)] = 1.0
    masks.append(mask)

    # 6: Pentagon (frog)
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, 6)[1:]
    verts = [np.array([c + r * math.cos(a), c + r * math.sin(a)]) for a in angles]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[_points_in_convex_polygon(pts, verts)] = 1.0
    masks.append(mask)

    # 7: Hexagon (horse)
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, 7)[1:]
    verts = [np.array([c + r * math.cos(a), c + r * math.sin(a)]) for a in angles]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[_points_in_convex_polygon(pts, verts)] = 1.0
    masks.append(mask)

    # 8: Ring / annulus (ship)
    mask = np.zeros((h, w), dtype=np.float32)
    outer = (xs - c) ** 2 + (ys - c) ** 2 <= r ** 2
    inner = (xs - c) ** 2 + (ys - c) ** 2 <= rr ** 2
    mask[outer & ~inner] = 1.0
    masks.append(mask)

    # 9: Horizontal rectangle (truck)
    mask = np.zeros((h, w), dtype=np.float32)
    rect_h = hw
    rect_w = r + hw
    mask[c - rect_h:c + rect_h, c - rect_w:c + rect_w] = 1.0
    masks.append(mask)

    return np.stack(masks, axis=0)  # (10, H, W)


SHAPE_MASKS = _generate_shape_masks(IMG_SIZE)

# Per-class background colours (RGB, 0-1 range) for synthetic images
CLASS_COLORS = np.array([
    [0.000, 0.447, 0.741],  # 0 airplane     - blue
    [0.850, 0.325, 0.098],  # 1 automobile   - orange
    [0.929, 0.694, 0.125],  # 2 bird         - yellow
    [0.494, 0.184, 0.556],  # 3 cat          - purple
    [0.466, 0.674, 0.188],  # 4 deer         - green
    [0.301, 0.745, 0.933],  # 5 dog          - cyan
    [0.635, 0.078, 0.184],  # 6 frog         - dark red
    [0.910, 0.573, 0.247],  # 7 horse        - peach
    [0.340, 0.157, 0.580],  # 8 ship         - indigo
    [0.580, 0.580, 0.580],  # 9 truck        - grey
], dtype=np.float32)


def generate_synthetic_dataset(seed=SEED, train=True):
    """Generate a completely synthetic dataset with RGB images and masks.

    Each image is a plain colored background (per class) with a
    randomly placed (small jitter) version of the shape mask.

    Returns:
        images: (N, 3, H, W) float32 tensor, values in ~[0, 1]
        masks:  (N, 1, H, W) float32 tensor, binary {0, 1}
        labels: (N,) int64 tensor
    """
    rng = np.random.default_rng(seed)
    n_per_class = NUM_TRAIN_PER_CLASS if train else NUM_TEST_PER_CLASS
    n_total = n_per_class * 10

    all_images = np.zeros((n_total, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    all_masks  = np.zeros((n_total, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    all_labels = np.zeros(n_total, dtype=np.int64)

    idx = 0
    for cls in range(10):
        color = CLASS_COLORS[cls]  # (3,)
        base_mask = SHAPE_MASKS[cls]  # (H, W)

        for i in range(n_per_class):
            # Background with slight per-sample noise
            img = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            for ch in range(3):
                noise = (rng.random((IMG_SIZE, IMG_SIZE)).astype(np.float32) - 0.5) * 0.05
                img[ch] = np.clip(color[ch] + noise, 0.0, 1.0)

            # Add jitter to the shape position (max 3 pixels shift)
            jx = rng.integers(-3, 4)
            jy = rng.integers(-3, 4)
            shape = np.roll(np.roll(base_mask, jy, axis=0), jx, axis=1)

            # Brighten the shape region in the image
            for ch in range(3):
                shape_ch = np.clip(color[ch] + 0.3 + (rng.random() - 0.5) * 0.1, 0.0, 1.0)
                img[ch, shape > 0.5] = shape_ch

            all_images[idx] = img
            all_masks[idx, 0] = shape
            all_labels[idx] = cls
            idx += 1

    return (torch.from_numpy(all_images),
            torch.from_numpy(all_masks),
            torch.from_numpy(all_labels))


class SyntheticCIFAR10(Dataset):
    """Synthetic CIFAR-10-like dataset with foreground shapes and bg colours."""

    def __init__(self, classes=None, train=True, seed=SEED):
        images, masks, labels = generate_synthetic_dataset(seed, train)

        if classes is not None:
            class_set = set(classes)
            keep = [i for i in range(len(labels)) if labels[i].item() in class_set]
            self.images = images[keep]
            self.masks  = masks[keep]
            self.labels = labels[keep]
        else:
            self.images = images
            self.masks  = masks
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.labels[idx]


def get_task_loaders(classes, train=True, batch_size=BATCH_SIZE):
    """Get DataLoader for a specific set of classes."""
    dataset = SyntheticCIFAR10(classes=classes, train=train)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=NUM_WORKERS,
    )


def get_joint_loaders(batch_size=BATCH_SIZE):
    """Get DataLoaders for ALL classes."""
    train_ds = SyntheticCIFAR10(classes=None, train=True)
    test_ds  = SyntheticCIFAR10(classes=None, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS)
    return train_loader, test_loader


# ═════════════════════════════════════════════════════════════════════════
#  2.  Model: Simple 3-level U-Net
# ═════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) x 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    """3-level encoder-decoder with skip connections.

    Input:   (B, 3, 32, 32)
    Output:  (B, 1, 32, 32)  -- logits for binary segmentation
    """

    def __init__(self, in_channels=3):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # 128 from up + 128 skip

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)     # 64 from up + 64 skip

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # (B, 64, 32, 32)
        p1 = self.pool1(e1)     # (B, 64, 16, 16)
        e2 = self.enc2(p1)      # (B, 128, 16, 16)
        p2 = self.pool2(e2)     # (B, 128, 8, 8)
        e3 = self.enc3(p2)      # (B, 256, 8, 8)

        # Decoder with skip connections
        d2_up = self.up2(e3)                      # (B, 128, 16, 16)
        d2 = torch.cat([d2_up, e2], dim=1)         # (B, 256, 16, 16)
        d2 = self.dec2(d2)                         # (B, 128, 16, 16)

        d1_up = self.up1(d2)                       # (B, 64, 32, 32)
        d1 = torch.cat([d1_up, e1], dim=1)         # (B, 128, 32, 32)
        d1 = self.dec1(d1)                         # (B, 64, 32, 32)

        out = self.out(d1)                         # (B, 1, 32, 32)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════
#  3.  Training / Evaluation Utilities
# ═════════════════════════════════════════════════════════════════════════

def binary_iou(pred_mask, gt_mask, eps=1e-6):
    """Compute binary IoU.

    pred_mask: (B, 1, H, W) logits
    gt_mask:   (B, 1, H, W) {0, 1}
    Returns: scalar IoU averaged over batch.
    """
    pred = torch.sigmoid(pred_mask)
    pred_bin = (pred > 0.5).float()
    gt = gt_mask.float()

    intersection = (pred_bin * gt).sum(dim=(1, 2, 3))
    union = (pred_bin + gt).clamp(0, 1).sum(dim=(1, 2, 3))
    iou_per = (intersection + eps) / (union + eps)
    return iou_per.mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute mean binary IoU over a dataloader."""
    model.eval()
    ious = []
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        iou = binary_iou(logits, masks)
        ious.append(iou)
    return np.mean(ious)


@torch.no_grad()
def evaluate_per_class(model, device, classes, batch_size=BATCH_SIZE):
    """Compute per-class IoU on the test set.

    Returns: dict {class_index: mean_iou}
    """
    model.eval()
    dataset = SyntheticCIFAR10(classes=classes, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS)

    class_ious = {c: [] for c in classes}
    for images, masks, labels in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        pred = torch.sigmoid(logits)
        pred_bin = (pred > 0.5).float()
        gt = masks.float()

        intersection = (pred_bin * gt).sum(dim=(1, 2, 3))
        union = (pred_bin + gt).clamp(0, 1).sum(dim=(1, 2, 3))
        iou_per = ((intersection + 1e-6) / (union + 1e-6)).cpu().numpy()

        for i, lbl in enumerate(labels.cpu().numpy()):
            class_ious[lbl].append(iou_per[i])
    return {c: float(np.mean(v)) for c, v in class_ious.items()}


# ═════════════════════════════════════════════════════════════════════════
#  4.  Distillation & EWC Utilities
# ═════════════════════════════════════════════════════════════════════════

def distillation_loss(student_logits, teacher_logits, temperature=TEMPERATURE):
    """KL-divergence distillation loss for binary segmentation.

    Treats each pixel as a 2-class softmax (fg vs bg).
    L_distill = KL(softmax(t/T) || softmax(s/T)) * T^2
    """
    T = temperature
    s_fg = F.logsigmoid(student_logits / T)
    s_bg = F.logsigmoid(-student_logits / T)
    t_fg = torch.sigmoid(teacher_logits / T)
    t_bg = torch.sigmoid(-teacher_logits / T)

    loss = t_fg * (t_fg.log() - s_fg) + t_bg * (t_bg.log() - s_bg)
    loss = loss.mean() * (T ** 2)
    return loss


def is_backbone_param(name):
    """Exclude the output convolution from EWC regularisation."""
    return "out" not in name


def compute_fisher(model, loader, device):
    """Compute diagonal Fisher information matrix.

    F_i = (1/N) * sum_n (dL/dtheta_i)^2
    """
    model.eval()
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and is_backbone_param(name):
            fisher_dict[name] = torch.zeros_like(param.data)

    criterion = nn.BCEWithLogitsLoss()
    n_samples = 0

    for images, masks, _ in tqdm(loader, desc="  Fisher", leave=False):
        images, masks = images.to(device), masks.to(device)
        model.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()

        bs = images.size(0)
        for name, param in model.named_parameters():
            if param.requires_grad and is_backbone_param(name) and param.grad is not None:
                fisher_dict[name] += param.grad.detach() ** 2 * bs
        n_samples += bs

    for name in fisher_dict:
        fisher_dict[name] /= n_samples
    return fisher_dict


def ewc_penalty(model, fisher_dict, optimal_params, lambda_ewc, device):
    """Compute EWC regularisation: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2."""
    loss = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in fisher_dict and name in optimal_params:
            if param.data.shape == optimal_params[name].shape:
                f = fisher_dict[name].to(device)
                theta_star = optimal_params[name].to(device)
                loss += torch.sum(f * (param - theta_star) ** 2)
    return (lambda_ewc / 2.0) * loss


# ═════════════════════════════════════════════════════════════════════════
#  5.  Experiment Methods
# ═════════════════════════════════════════════════════════════════════════

def train_one_epoch_seg(model, loader, optimizer, criterion, device,
                        ewc_fn=None, distill_loader=None, teacher_model=None):
    """Train for one epoch. Returns (avg_bce_loss, avg_iou)."""
    model.train()
    running_bce = 0.0
    running_iou = 0.0
    n_batches = 0

    distill_iter = None
    if teacher_model is not None and distill_loader is not None:
        distill_iter = iter(distill_loader)

    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        # Distillation (LwF)
        if teacher_model is not None and distill_iter is not None:
            try:
                d_img, _, _ = next(distill_iter)
            except StopIteration:
                distill_iter = iter(distill_loader)
                d_img, _, _ = next(distill_iter)
            d_img = d_img.to(device)
            with torch.no_grad():
                t_logits = teacher_model(d_img)
            s_logits = model(d_img)
            loss += LAMBDA_DISTILL * distillation_loss(s_logits, t_logits)

        # EWC
        if ewc_fn is not None:
            loss += ewc_fn(model)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            iou = binary_iou(logits.detach(), masks)

        running_bce += loss.item()
        running_iou += iou
        n_batches += 1

    return running_bce / n_batches, running_iou / n_batches


def train_model(model, train_loader, num_epochs=TRAIN_EPOCHS, device=DEVICE,
                ewc_fn=None, distill_loader=None, teacher_model=None,
                desc="Training"):
    """Generic training loop. Returns (epoch_losses, epoch_ious)."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    epoch_losses = []
    epoch_ious = []

    for epoch in range(1, num_epochs + 1):
        bce, iou = train_one_epoch_seg(
            model, train_loader, optimizer, criterion, device,
            ewc_fn=ewc_fn, distill_loader=distill_loader,
            teacher_model=teacher_model,
        )
        epoch_losses.append(bce)
        epoch_ious.append(iou)

        if epoch == 1 or epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}/{num_epochs}  BCE: {bce:.4f}  IoU: {iou:.4f}")

    return epoch_losses, epoch_ious


# ──────────── 5a. Fine-tune  ────────────

def run_finetune():
    """Sequential fine-tune: Task A -> Task B."""
    print("\n" + "=" * 60)
    print("  METHOD: Fine-tune")
    print("=" * 60)

    model = SimpleUNet(in_channels=3).to(DEVICE)
    print(f"  Model params: {count_params(model):,}")

    # --- Task A ---
    print("\n--- Task A (classes 0-4) ---")
    loader_a = get_task_loaders(TASK_A_CLASSES, train=True)
    train_model(model, loader_a, desc="Fine-tune Task A")

    pc_a_after_a = evaluate_per_class(model, DEVICE, TASK_A_CLASSES)

    # --- Task B ---
    print("\n--- Task B (classes 5-9) ---")
    loader_b = get_task_loaders(TASK_B_CLASSES, train=True)
    train_model(model, loader_b, desc="Fine-tune Task B")

    pc_a_final = evaluate_per_class(model, DEVICE, TASK_A_CLASSES)
    pc_b_final = evaluate_per_class(model, DEVICE, TASK_B_CLASSES)

    return model, {
        "pc_a_after_a": pc_a_after_a,
        "pc_a_final": pc_a_final,
        "pc_b_final": pc_b_final,
    }


# ──────────── 5b. LwF  ────────────

def run_lwf():
    """Learning without Forgetting: distillation on Task A data."""
    print("\n" + "=" * 60)
    print("  METHOD: LwF (Learning without Forgetting)")
    print("=" * 60)

    model = SimpleUNet(in_channels=3).to(DEVICE)
    print(f"  Model params: {count_params(model):,}")

    # --- Task A ---
    print("\n--- Task A (classes 0-4) ---")
    loader_a = get_task_loaders(TASK_A_CLASSES, train=True)
    train_model(model, loader_a, desc="LwF Task A")

    pc_a_after_a = evaluate_per_class(model, DEVICE, TASK_A_CLASSES)

    # Save teacher
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print("  Teacher frozen")

    # --- Task B with distillation ---
    print("\n--- Task B (classes 5-9) with distillation ---")
    loader_b     = get_task_loaders(TASK_B_CLASSES, train=True)
    loader_dist  = get_task_loaders(TASK_A_CLASSES, train=True)

    train_model(model, loader_b, teacher_model=teacher,
                distill_loader=loader_dist, desc="LwF Task B")

    pc_a_final = evaluate_per_class(model, DEVICE, TASK_A_CLASSES)
    pc_b_final = evaluate_per_class(model, DEVICE, TASK_B_CLASSES)

    return model, {
        "pc_a_after_a": pc_a_after_a,
        "pc_a_final": pc_a_final,
        "pc_b_final": pc_b_final,
    }


# ──────────── 5c. EWC  ────────────

def run_ewc():
    """Elastic Weight Consolidation."""
    print("\n" + "=" * 60)
    print(f"  METHOD: EWC (lambda = {EWC_LAMBDA})")
    print("=" * 60)

    model = SimpleUNet(in_channels=3).to(DEVICE)
    print(f"  Model params: {count_params(model):,}")

    # --- Task A ---
    print("\n--- Task A (classes 0-4) ---")
    loader_a = get_task_loaders(TASK_A_CLASSES, train=True)
    train_model(model, loader_a, desc="EWC Task A")

    pc_a_after_a = evaluate_per_class(model, DEVICE, TASK_A_CLASSES)

    # Compute Fisher (subsampled for speed)
    n_fisher = min(FISHER_SUBSAMPLE, len(SyntheticCIFAR10(classes=TASK_A_CLASSES, train=True)))
    subset_indices = list(range(n_fisher))
    fisher_dataset = SyntheticCIFAR10(classes=TASK_A_CLASSES, train=True)
    fisher_subset = torch.utils.data.Subset(fisher_dataset, subset_indices)
    fisher_loader = DataLoader(fisher_subset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS)
    print(f"  Fisher using {n_fisher} samples")
    fisher_dict = compute_fisher(model, fisher_loader, DEVICE)
    optimal_params = {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad and is_backbone_param(name)
    }
    print("  Fisher computed")

    # --- Task B with EWC ---
    print("\n--- Task B (classes 5-9) with EWC ---")
    loader_b = get_task_loaders(TASK_B_CLASSES, train=True)

    def ewc_fn(m):
        return ewc_penalty(m, fisher_dict, optimal_params, EWC_LAMBDA, DEVICE)

    train_model(model, loader_b, ewc_fn=ewc_fn, desc="EWC Task B")

    pc_a_final = evaluate_per_class(model, DEVICE, TASK_A_CLASSES)
    pc_b_final = evaluate_per_class(model, DEVICE, TASK_B_CLASSES)

    return model, {
        "pc_a_after_a": pc_a_after_a,
        "pc_a_final": pc_a_final,
        "pc_b_final": pc_b_final,
    }


# ──────────── 5d. Joint (upper bound)  ────────────

def run_joint():
    """Joint training on all 10 classes simultaneously (upper bound)."""
    print("\n" + "=" * 60)
    print("  METHOD: Joint (upper bound)")
    print("=" * 60)

    model = SimpleUNet(in_channels=3).to(DEVICE)
    print(f"  Model params: {count_params(model):,}")

    train_loader, test_loader = get_joint_loaders()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    epoch_ious = []

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        n_batches = 0

        for images, masks, _ in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_iou += binary_iou(logits.detach(), masks)
            n_batches += 1

        avg_iou = running_iou / n_batches
        epoch_ious.append(avg_iou)
        if epoch == 1 or epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}/{TRAIN_EPOCHS}  Loss: {running_loss / n_batches:.4f}  IoU: {avg_iou:.4f}")

    pc_a = evaluate_per_class(model, DEVICE, TASK_A_CLASSES)
    pc_b = evaluate_per_class(model, DEVICE, TASK_B_CLASSES)

    return model, {
        "pc_a_final": pc_a,
        "pc_b_final": pc_b,
        "epoch_ious": epoch_ious,
    }


# ═════════════════════════════════════════════════════════════════════════
#  6.  Visualisation & Output
# ═════════════════════════════════════════════════════════════════════════

def save_per_class_iou_table(all_results, save_dir):
    """Save per-class IoU comparison as CSV and print a formatted table."""
    classes = TASK_A_CLASSES + TASK_B_CLASSES
    methods = list(all_results.keys())

    rows = []
    for c in classes:
        row = {"class": f"{c} ({CIFAR10_NAMES[c]})"}
        for m in methods:
            if m == "Joint":
                pc = all_results[m]["pc_a_final"] if c < 5 else all_results[m]["pc_b_final"]
                row[f"{m}_IoU"] = pc.get(c, 0)
            else:
                row[f"{m}_IoU"] = all_results[m]["pc_a_final"].get(
                    c, all_results[m]["pc_b_final"].get(c, 0))
        rows.append(row)

    # CSV
    csv_path = os.path.join(save_dir, "per_class_iou.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {csv_path}")

    # Print table
    print("\n  Per-class IoU (final after both tasks):")
    header = f"  {'Class':20s}"
    for m in methods:
        header += f"  {m:10s}"
    print(header)
    print("  " + "-" * (21 + 13 * len(methods)))
    for c in classes:
        name = f"{c} ({CIFAR10_NAMES[c]})"
        line = f"  {name:20s}"
        for m in methods:
            if m == "Joint":
                pc = all_results[m]["pc_a_final"] if c < 5 else all_results[m]["pc_b_final"]
                v = pc.get(c, 0)
            else:
                pc = all_results[m]["pc_a_final"] if c < 5 else all_results[m]["pc_b_final"]
                v = pc.get(c, 0)
            line += f"  {v:.4f}    "
        print(line)
    print("  " + "-" * (21 + 13 * len(methods)))

    # Group averages
    for group_name, group in [("Task A (0-4)", TASK_A_CLASSES),
                               ("Task B (5-9)", TASK_B_CLASSES)]:
        print(f"\n  Mean {group_name}:")
        for m in methods:
            pc = all_results[m]["pc_a_final"] if group_name.startswith("Task A") else all_results[m]["pc_b_final"]
            avg = np.mean([pc.get(c, 0) for c in group])
            print(f"    {m:12s}: {avg:.4f}")

    return csv_path


def plot_forgetting_rate(all_results, save_dir):
    """Bar chart of forgetting (IoU drop) on Task A classes."""
    methods = ["Fine-tune", "LwF", "EWC"]
    classes = TASK_A_CLASSES
    class_names = [f"{c}\n{CIFAR10_NAMES[c]}" for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for idx, method in enumerate(methods):
        forgetting = [
            all_results[method]["pc_a_after_a"].get(c, 0)
            - all_results[method]["pc_a_final"].get(c, 0)
            for c in classes
        ]
        bars = ax.bar(x + idx * width, forgetting, width,
                      label=method, color=colors[idx], alpha=0.85)
        for bar, val in zip(bars, forgetting):
            if abs(val) > 0.005:
                y_pos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.025
                va = "bottom" if val >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f"{val:.3f}", ha="center", va=va, fontsize=8)

    ax.set_xlabel("Class")
    ax.set_ylabel("Forgetting (IoU drop)")
    ax.set_title("Forgetting on Task A after Task B Training")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "forgetting_rate.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_seg_visualization(all_results, save_dir):
    """Side-by-side: image | GT | Fine-tune | LwF | EWC | Joint."""
    method_names = ["Fine-tune", "LwF", "EWC", "Joint"]
    n_methods = len(method_names) + 2  # image + GT

    # 4 samples: 2 from Task A (classes 0, 2), 2 from Task B (classes 5, 7)
    test_ds = SyntheticCIFAR10(classes=None, train=False)
    samples = []
    for cls in [0, 2, 5, 7]:
        # find first sample of that class
        for i in range(len(test_ds)):
            if test_ds.labels[i].item() == cls:
                samples.append((i, cls))
                break

    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, n_methods,
                             figsize=(3 * n_methods, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, (idx, cls) in enumerate(samples):
        img, gt_mask, label = test_ds[idx]

        # Denormalise: values are in [0, 1] since synthetic
        img_disp = img.permute(1, 2, 0).numpy()

        # Column 0: original
        axes[row, 0].imshow(np.clip(img_disp, 0, 1))
        axes[row, 0].set_title(f"Input\n({CIFAR10_NAMES[cls]})", fontsize=9)
        axes[row, 0].axis("off")

        # Column 1: GT
        axes[row, 1].imshow(gt_mask.squeeze().numpy(), cmap="Blues", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth", fontsize=9)
        axes[row, 1].axis("off")

        # Columns 2+: predictions
        for col, method in enumerate(method_names):
            model = all_results[method].get("model")
            if model is None:
                axes[row, col + 2].text(0.5, 0.5, "N/A", ha="center", va="center")
                axes[row, col + 2].axis("off")
                continue

            model.eval()
            with torch.no_grad():
                logits = model(img.unsqueeze(0).to(DEVICE))
                pred = torch.sigmoid(logits).squeeze().cpu().numpy()
            axes[row, col + 2].imshow(pred, cmap="Blues", vmin=0, vmax=1)
            axes[row, col + 2].set_title(method, fontsize=9)
            axes[row, col + 2].axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "seg_visualization.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_iou_progress(all_results, save_dir):
    """IoU vs training progress bar chart."""
    methods = ["Fine-tune", "LwF", "EWC", "Joint"]
    x = np.arange(len(methods))
    width = 0.3
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, method in enumerate(methods):
        info = all_results[method]
        if method == "Joint":
            a_val = np.mean(list(info["pc_a_final"].values()))
            b_val = np.mean(list(info["pc_b_final"].values()))
        else:
            a_val = np.mean(list(info["pc_a_final"].values()))
            b_val = np.mean(list(info["pc_b_final"].values()))
        ax.bar(x[idx] - width / 2, a_val, width, label=f"{method} Task A",
               color=colors[idx], alpha=0.7)
        ax.bar(x[idx] + width / 2, b_val, width, label=f"{method} Task B",
               color=colors[idx], alpha=0.3, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Mean IoU")
    ax.set_title("Final IoU by Method and Task")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "iou_vs_epochs.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def save_results_json(all_results, save_dir):
    """Save numeric results as JSON."""
    serializable = {}
    for method_name, info in all_results.items():
        entry = {}
        for k, v in info.items():
            if isinstance(v, dict):
                entry[k] = {str(kk): float(vv) for kk, vv in v.items()}
            elif isinstance(v, list):
                entry[k] = [float(x) for x in v]
            elif isinstance(v, nn.Module):
                continue
            else:
                entry[k] = v
        serializable[method_name] = entry

    path = os.path.join(save_dir, "all_results.json")
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════
#  7.  Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Experiment 8: Incremental Segmentation with Distillation")
    print(f"  Device: {DEVICE}")
    print(f"  Outputs: {OUTPUT_DIR}")
    print(f"  Data: synthetic CIFAR-10-like ({NUM_TRAIN_PER_CLASS} train, "
          f"{NUM_TEST_PER_CLASS} test per class)")
    print(f"  Epochs per task: {TRAIN_EPOCHS}")
    print(f"  Fisher samples: {FISHER_SUBSAMPLE}")
    print("=" * 60)

    # Visualise shape templates
    print("\n--- Shape Templates ---")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for cls in ALL_CLASSES:
        row, col = cls // 5, cls % 5
        axes[row, col].imshow(SHAPE_MASKS[cls], cmap="Blues", vmin=0, vmax=1)
        axes[row, col].set_title(f"{cls}: {CIFAR10_NAMES[cls]}")
        axes[row, col].axis("off")
    plt.tight_layout()
    shape_path = os.path.join(OUTPUT_DIR, "shape_templates.png")
    plt.savefig(shape_path, dpi=150)
    plt.close()
    print(f"  Saved {shape_path}")

    # Dataset statistics
    ds = SyntheticCIFAR10(classes=None, train=True)
    print(f"\n  Training samples: {len(ds)}")
    ds_te = SyntheticCIFAR10(classes=None, train=False)
    print(f"  Test samples:     {len(ds_te)}")

    # ── Run experiments ──────────────────────────────────────────
    all_results = {}

    # 1. Fine-tune
    set_seed()
    model_ft, info_ft = run_finetune()
    all_results["Fine-tune"] = info_ft
    all_results["Fine-tune"]["model"] = model_ft

    # 2. LwF
    set_seed()
    model_lwf, info_lwf = run_lwf()
    all_results["LwF"] = info_lwf
    all_results["LwF"]["model"] = model_lwf

    # 3. EWC
    set_seed()
    model_ewc, info_ewc = run_ewc()
    all_results["EWC"] = info_ewc
    all_results["EWC"]["model"] = model_ewc

    # 4. Joint
    set_seed()
    model_joint, info_joint = run_joint()
    all_results["Joint"] = info_joint
    all_results["Joint"]["model"] = model_joint

    # ── Visualisation & Output ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating Outputs")
    print("=" * 60)

    save_per_class_iou_table(all_results, OUTPUT_DIR)
    plot_forgetting_rate(all_results, OUTPUT_DIR)
    plot_seg_visualization(all_results, OUTPUT_DIR)
    plot_iou_progress(all_results, OUTPUT_DIR)
    save_results_json(all_results, OUTPUT_DIR)

    # ── Final Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for method in ["Fine-tune", "LwF", "EWC", "Joint"]:
        info = all_results[method]
        avg_a = np.mean(list(info["pc_a_final"].values()))
        avg_b = np.mean(list(info["pc_b_final"].values()))
        print(f"  {method:12s}: Task A IoU = {avg_a:.4f}, Task B IoU = {avg_b:.4f}")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
