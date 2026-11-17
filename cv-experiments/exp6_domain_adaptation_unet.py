#!/usr/bin/env python3
"""
Experiment 6: Domain Adaptation for Segmentation using U-Net with
Gradient Reversal Layer (DANN-style) on synthetic vessel-like data.

Data:
  - Synthetic fundus-like images with random ellipse / curve shapes as "vessels"
  - Source domain: one color palette (bright background)
  - Target domain: different color palette + GaussianBlur (domain shift)
  - No external downloads required.

Architecture:
  - U-Net (3-level encoder-decoder with skip connections)
  - Domain Discriminator: 3-layer MLP (256->128->64->1)
  - Gradient Reversal Layer (GRL) between encoder and discriminator

Training paradigms compared:
  1. Source-only: standard U-Net on source domain (no adaptation)
  2. Adversarial: U-Net encoder + discriminator (DANN-style adaptation)

Outputs (saved to outputs/exp6/):
  - Target-domain Dice comparison bar chart
  - Segmentation visualizations (before vs after adaptation)
  - t-SNE of encoder bottleneck features
  - Training curves, results JSON
"""

import os, sys, copy, json, time, warnings, random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE

from scipy.ndimage import label as _label, gaussian_filter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs", "exp6")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-3
LAMBDA_ADV = 0.01           # GRL scaling factor for adversarial loss
NUM_SAMPLES = 2000          # training samples per domain
NUM_TEST_SAMPLES = 400      # test samples per domain
NUM_WORKERS = 0             # 0 for Windows compatibility
SEED = 42
IMG_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ═══════════════════════════════════════════════════════════════════════════
# 1. Synthetic Vessel Data Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_vessel_mask(size=32, num_ellipses=8, thickness_range=(1, 3)):
    """Generate a binary mask with random ellipse/curve shapes as "vessels".

    Returns:
        mask: (size, size) float32 array, values in {0, 1}
    """
    mask = np.zeros((size, size), dtype=np.float32)

    for _ in range(num_ellipses):
        # Random ellipse parameters
        cx = random.uniform(0.1, 0.9) * size
        cy = random.uniform(0.1, 0.9) * size
        a = random.uniform(0.05, 0.25) * size   # semi-major axis
        b = random.uniform(0.02, 0.10) * size   # semi-minor axis (thin vessel)
        angle = random.uniform(0, np.pi)         # rotation

        # Slight curvature: add a third ellipse offset for Y-shaped / curved vessels
        thickness = random.randint(*thickness_range)

        # Create ellipse
        yy, xx = np.ogrid[:size, :size]
        xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)

        # Ellipse equation
        ellipse = (xr ** 2) / (a ** 2) + (yr ** 2) / (b ** 2)
        vessel = (ellipse <= 1.0).astype(np.float32)

        # Apply thickness (dilate slightly)
        if thickness > 1:
            vessel = gaussian_filter(vessel, sigma=thickness * 0.4, mode="constant")
            vessel = (vessel > 0.3).astype(np.float32)

        mask = np.clip(mask + vessel, 0, 1)

        # Sometimes add a branch (Y-shape)
        if random.random() < 0.4:
            cx2 = cx + random.uniform(-0.15, 0.15) * size
            cy2 = cy + random.uniform(-0.15, 0.15) * size
            a2 = random.uniform(0.04, 0.18) * size
            b2 = random.uniform(0.02, 0.06) * size
            angle2 = angle + random.uniform(-0.5, 0.5)
            xr2 = (xx - cx2) * np.cos(angle2) + (yy - cy2) * np.sin(angle2)
            yr2 = -(xx - cx2) * np.sin(angle2) + (yy - cy2) * np.cos(angle2)
            ellipse2 = (xr2 ** 2) / (a2 ** 2) + (yr2 ** 2) / (b2 ** 2)
            branch = (ellipse2 <= 1.0).astype(np.float32)
            mask = np.clip(mask + branch, 0, 1)

    # Keep only the largest connected component (optional, for realism)
    labelled, n_feats = _label(mask)
    if n_feats > 1:
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        largest = np.argmax(sizes)
        mask = (labelled == largest).astype(np.float32)

    return mask


def render_vessel_image(mask, source_domain=True):
    """Render a binary vessel mask as a colour image.

    Source domain:
        Background:     light cream (210, 205, 190) → same as target but brighter
        Vessels:        dark reddish-brown (90, 35, 35)
        Added:          subtle noise

    Target domain:
        Background:     darker blue-grey (70, 80, 105)
        Vessels:        brighter reddish (190, 80, 60)
        Added:          stronger noise + mild Gaussian blur

    Note: both domains share the same vessel morphology; only the colour
          palette and noise level differ. The background colour difference
          is the main domain shift.
    """
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    if source_domain:
        bg_colour = np.array([210, 205, 190], dtype=np.float32)
        vessel_colour = np.array([90, 35, 35], dtype=np.float32)
        noise_std = 4.0
    else:
        bg_colour = np.array([70, 80, 105], dtype=np.float32)
        vessel_colour = np.array([190, 80, 60], dtype=np.float32)
        noise_std = 15.0

    # Composite background + vessels
    for c in range(3):
        img[:, :, c] = bg_colour[c] * (1 - mask) + vessel_colour[c] * mask

    # Add Gaussian noise
    img += np.random.randn(h, w, 3) * noise_std

    # Target domain gets mild Gaussian blur
    if not source_domain:
        from scipy.ndimage import gaussian_filter as _gf
        for c in range(3):
            img[:, :, c] = _gf(img[:, :, c], sigma=0.8)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


class SyntheticVesselDataset(Dataset):
    """Self-contained synthetic vessel dataset.

    Args:
        num_samples:   number of images to generate
        source_domain: True → source colour palette; False → target palette
        transform:     optional torchvision transform
    """

    def __init__(self, num_samples, source_domain=True, transform=None):
        self.num_samples = num_samples
        self.source_domain = source_domain
        self.transform = transform or transforms.ToTensor()

        # Pre-generate all masks and images deterministically
        rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        random.seed(SEED + (0 if source_domain else 10000))
        np.random.seed(SEED + (0 if source_domain else 10000))

        print(f"  Generating {num_samples} {'source' if source_domain else 'target'} "
              f"samples ...")
        self.images = []
        self.masks = []
        for i in tqdm(range(num_samples), leave=False):
            mask = generate_vessel_mask(size=IMG_SIZE)
            img_np = render_vessel_image(mask, source_domain=source_domain)

            self.masks.append(torch.from_numpy(mask))
            self.images.append(img_np)

        # Restore RNG state
        random.setstate(rng_state)
        np.random.set_state(np_rng_state)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_np = self.images[idx]       # (H, W, 3)  uint8
        mask = self.masks[idx]           # (H, W)     float32

        # Apply transform (ToTensor + Normalize)
        img_tensor = self.transform(img_np)

        # Return dummy label 0 (kept for API compatibility)
        return img_tensor, mask, 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. U-Net  (3-level encoder-decoder)
# ═══════════════════════════════════════════════════════════════════════════

class UNet(nn.Module):
    """Simple 3-level U-Net for 32 x 32 inputs.

    Encoder:
        Conv(3→64)→Conv(64→64)→Pool  →  Conv(64→128)→Conv(128→128)→Pool
      → Conv(128→256)→Conv(256→256)→Pool

    Decoder:
        Up→Conv(256→128)→Up→Conv(128→64)→Up→Conv(64→1)
    with skip connections from each encoder level.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        # ---- Encoder ----
        # Level 1 :  3 → 64
        self.enc1_conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc1_bn1   = nn.BatchNorm2d(64)
        self.enc1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.enc1_bn2   = nn.BatchNorm2d(64)

        # Level 2 : 64 → 128
        self.enc2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc2_bn1   = nn.BatchNorm2d(128)
        self.enc2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc2_bn2   = nn.BatchNorm2d(128)

        # Level 3 (bottleneck) : 128 → 256
        self.enc3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc3_bn1   = nn.BatchNorm2d(256)
        self.enc3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.enc3_bn2   = nn.BatchNorm2d(256)

        # ---- Decoder ----
        #
        # Skip connections (features BEFORE each pool):
        #   e1 (32×32,  64 ch)  → skip for up3
        #   e2 (16×16, 128 ch)  → skip for up2
        #   e3 ( 8×8,  256 ch)  → skip for up1
        #
        # Up1 : p3 (4×4) → 8×8  →  concat e3 (256 ch)  →  Conv(512→128)
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec1_conv = nn.Conv2d(256 + 256, 128, 3, padding=1)
        self.dec1_bn   = nn.BatchNorm2d(128)

        # Up2 : 8×8 → 16×16  →  concat e2 (128 ch)  →  Conv(256→64)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2_conv = nn.Conv2d(128 + 128, 64, 3, padding=1)
        self.dec2_bn   = nn.BatchNorm2d(64)

        # Up3 : 16×16 → 32×32  →  concat e1 (64 ch)  →  Conv(128→1)
        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec3_conv = nn.Conv2d(64 + 64, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ---- Encoder ----
        e1 = F.relu(self.enc1_bn1(self.enc1_conv1(x)))
        e1 = F.relu(self.enc1_bn2(self.enc1_conv2(e1)))     # (B, 64, 32, 32)  skip1
        p1 = F.max_pool2d(e1, 2)                             # (B, 64, 16, 16)

        e2 = F.relu(self.enc2_bn1(self.enc2_conv1(p1)))
        e2 = F.relu(self.enc2_bn2(self.enc2_conv2(e2)))     # (B,128, 16, 16)  skip2
        p2 = F.max_pool2d(e2, 2)                             # (B,128,  8,  8)

        e3 = F.relu(self.enc3_bn1(self.enc3_conv1(p2)))
        e3 = F.relu(self.enc3_bn2(self.enc3_conv2(e3)))     # (B,256,  8,  8)  bottleneck
        p3 = F.max_pool2d(e3, 2)                             # (B,256,  4,  4)

        # ---- Decoder ----
        d1 = self.up1(p3)                                    # (B,256,  8,  8)
        d1 = torch.cat([d1, e3], dim=1)                      # (B,512,  8,  8)  concat e3
        d1 = F.relu(self.dec1_bn(self.dec1_conv(d1)))        # (B,128,  8,  8)

        d2 = self.up2(d1)                                    # (B,128, 16, 16)
        d2 = torch.cat([d2, e2], dim=1)                      # (B,256, 16, 16)  concat e2
        d2 = F.relu(self.dec2_bn(self.dec2_conv(d2)))        # (B, 64, 16, 16)

        d3 = self.up3(d2)                                    # (B, 64, 32, 32)
        d3 = torch.cat([d3, e1], dim=1)                      # (B,128, 32, 32)  concat e1
        out = self.dec3_conv(d3)                             # (B,  1, 32, 32)

        return out, e3                                       # logits,  bottleneck


# ═══════════════════════════════════════════════════════════════════════════
# 3. Gradient Reversal Layer  &  Domain Discriminator
# ═══════════════════════════════════════════════════════════════════════════

class GradientReversalFunction(torch.autograd.Function):
    """Forward: identity.  Backward: multiply gradient by -lambda."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Thin wrapper around GradientReversalFunction."""

    def __init__(self, lambda_=0.1):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """3-layer MLP: 256 → 128 → 64 → 1  (logit output)."""

    def __init__(self, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)                                   # (B, 1) logits


# ═══════════════════════════════════════════════════════════════════════════
# 4. Full Domain-Adaptation U-Net
# ═══════════════════════════════════════════════════════════════════════════

class DomainAdaptationUNet(nn.Module):
    """U-Net segmenter + domain discriminator with GRL."""

    def __init__(self, lambda_adv=0.1):
        super().__init__()
        self.unet          = UNet(in_channels=3)
        self.discriminator = DomainDiscriminator(feat_dim=256)
        self.grl           = GradientReversalLayer(lambda_=lambda_adv)
        self.gap           = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Return (seg_logits, bottleneck_feature_map)."""
        return self.unet(x)

    def get_features(self, x):
        """Return (seg_logits, 256-dim GAP-pooled feature vector)."""
        seg_out, bottleneck = self.unet(x)
        feat = self.gap(bottleneck).view(bottleneck.size(0), -1)
        return seg_out, feat

    def get_domain_pred(self, bottleneck):
        """Return domain logit from bottleneck features (with GRL applied)."""
        feat = self.gap(bottleneck).view(bottleneck.size(0), -1)
        feat_rev = self.grl(feat)
        return self.discriminator(feat_rev)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Training Utilities
# ═══════════════════════════════════════════════════════════════════════════

def dice_coefficient(pred, target, eps=1e-7):
    """Dice coefficient for binary masks.

    *pred*  – probability map (after sigmoid)  (B, 1, H, W)
    *target* – binary ground truth               (B, 1, H, W)
    """
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2.0 * intersection + eps) / (union + eps)


def train_source_only(model, loader, num_epochs, device):
    """Phase 1: train U-Net on source domain only (standard segmentation)."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    bce_loss = nn.BCEWithLogitsLoss()

    history = {"loss": [], "dice": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n = 0

        pbar = tqdm(loader, desc=f"Source-only  E{epoch:2d}/{num_epochs}",
                     leave=False)
        for imgs, masks, _ in pbar:
            imgs  = imgs.to(device)
            masks = masks.to(device).unsqueeze(1)            # (B, 1, 32, 32)

            optimizer.zero_grad()
            seg_out, _ = model(imgs)                         # logits
            loss = bce_loss(seg_out, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            with torch.no_grad():
                epoch_dice += dice_coefficient(torch.sigmoid(seg_out),
                                                masks).mean().item()
            n += 1

        avg_loss = epoch_loss / n
        avg_dice = epoch_dice / n
        history["loss"].append(avg_loss)
        history["dice"].append(avg_dice)

        tqdm.write(f"  Epoch {epoch:2d}/{num_epochs}  "
                   f"Loss: {avg_loss:.4f}  Dice: {avg_dice:.4f}")

    return history


def train_adversarial(model, src_loader, tgt_loader, num_epochs, device,
                      lambda_adv):
    """Phase 2: adversarial training with GRL and domain discriminator.

    Segmentation loss on source; domain loss on both source and target.
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)
    bce_loss = nn.BCEWithLogitsLoss()                       # segmentation
    domain_loss_fn = nn.BCEWithLogitsLoss()                  # domain

    history = {"seg_loss": [], "domain_loss": [], "dice": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_seg = 0.0
        epoch_dom = 0.0
        epoch_dice = 0.0
        n = 0

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        n_total = max(len(src_loader), len(tgt_loader))

        pbar = tqdm(range(n_total), desc=f"Adversarial E{epoch:2d}/{num_epochs}",
                     leave=False)
        for _ in pbar:
            # ---- Source batch ----
            try:
                s_img, s_mask, _ = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                s_img, s_mask, _ = next(src_iter)

            # ---- Target batch ----
            try:
                t_img, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                t_img, _, _ = next(tgt_iter)

            s_img  = s_img.to(device)
            s_mask = s_mask.to(device).unsqueeze(1)
            t_img  = t_img.to(device)

            optimizer.zero_grad()

            # --- Source: segmentation + domain ---
            seg_out, bottleneck_s = model(s_img)
            loss_seg = bce_loss(seg_out, s_mask)

            dom_pred_s = model.get_domain_pred(bottleneck_s)
            dom_label_s = torch.zeros_like(dom_pred_s)       # source → 0
            loss_dom = domain_loss_fn(dom_pred_s, dom_label_s)

            # --- Target: domain only ---
            _, bottleneck_t = model(t_img)
            dom_pred_t = model.get_domain_pred(bottleneck_t)
            dom_label_t = torch.ones_like(dom_pred_t)        # target → 1
            loss_dom += domain_loss_fn(dom_pred_t, dom_label_t)

            total_loss = loss_seg + lambda_adv * loss_dom
            total_loss.backward()
            optimizer.step()

            epoch_seg += loss_seg.item()
            epoch_dom += loss_dom.item()
            with torch.no_grad():
                epoch_dice += dice_coefficient(torch.sigmoid(seg_out),
                                                s_mask).mean().item()
            n += 1

        avg_seg  = epoch_seg / n
        avg_dom  = epoch_dom / n
        avg_dice = epoch_dice / n
        history["seg_loss"].append(avg_seg)
        history["domain_loss"].append(avg_dom)
        history["dice"].append(avg_dice)

        tqdm.write(f"  Epoch {epoch:2d}/{num_epochs}  "
                   f"Seg: {avg_seg:.4f}  Dom: {avg_dom:.4f}  Dice: {avg_dice:.4f}")

    return history


# ═══════════════════════════════════════════════════════════════════════════
# 6. Evaluation & Visualisation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    """Return mean Dice coefficient over *loader*."""
    model.eval()
    total_dice = 0.0
    n = 0
    for imgs, masks, _ in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device).unsqueeze(1)
        seg_out, _ = model(imgs)
        pred = torch.sigmoid(seg_out)
        total_dice += dice_coefficient(pred, masks).mean().item()
        n += 1
    return total_dice / n


@torch.no_grad()
def extract_features(model, loader, device, max_samples=1000):
    """Return (N, 256) array of GAP-pooled bottleneck features."""
    model.eval()
    feats = []
    count = 0
    for imgs, _, _ in loader:
        if count >= max_samples:
            break
        imgs = imgs.to(device)
        _, feat = model.get_features(imgs)
        feats.append(feat.cpu().numpy())
        count += imgs.size(0)
    return np.concatenate(feats, axis=0)[:max_samples]


def plot_dice_comparison(dice_before, dice_after, save_path):
    """Bar chart comparing Dice before and after adaptation."""
    labels = ["Source Only\n(Before Adapt.)", "Adversarial\n(After Adapt.)"]
    scores = [dice_before, dice_after]

    fig, ax = plt.subplots(figsize=(6, 5))
    colours = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, scores, color=colours, width=0.5, edgecolor="gray")

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{s:.4f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylabel("Dice Coefficient", fontsize=13)
    ax.set_title("Target Domain Segmentation", fontsize=14)
    ax.set_ylim(0, max(scores) * 1.25 + 0.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def visualize_predictions(model_before, model_after, loader, device,
                          num_samples=8, save_path=None):
    """Side-by-side comparison: input, GT, before-adapt, after-adapt."""
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "segmentation_comparison.png")

    model_before.eval()
    model_after.eval()

    batch = next(iter(loader))
    imgs = batch[0][:num_samples].to(device)
    masks = batch[1][:num_samples].to(device)

    with torch.no_grad():
        pred_before, _ = model_before(imgs)
        pred_after, _  = model_after(imgs)

    pred_before = torch.sigmoid(pred_before).cpu().numpy()
    pred_after  = torch.sigmoid(pred_after).cpu().numpy()
    imgs_np     = imgs.cpu().numpy()
    masks_np    = masks.cpu().numpy()

    n_rows = min(num_samples, 8)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 2.5 * n_rows))

    for i in range(n_rows):
        # Denormalise image (use per-channel stats from dataset)
        img = imgs_np[i].transpose(1, 2, 0)
        # Simple denormalisation with approximate stats
        img = img * 0.5 + 0.5   # undo approximate Normalise(0.5, 0.5)
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input (Target)", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks_np[i], cmap="gray")
        axes[i, 1].set_title("Ground Truth", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_before[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Before Adapt.", fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred_after[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 3].set_title("After Adapt.", fontsize=9)
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_tsne(feats_before_src, feats_before_tgt,
              feats_after_src, feats_after_tgt,
              save_path):
    """Two-panel t-SNE: encoder features before and after adaptation.

    Blue = source domain,  Red = target domain.
    """
    n = min(1000, feats_before_src.shape[0], feats_before_tgt.shape[0],
            feats_after_src.shape[0], feats_after_tgt.shape[0])

    # Stack for joint embedding
    combined = np.vstack([
        feats_before_src[:n], feats_before_tgt[:n],
        feats_after_src[:n],  feats_after_tgt[:n],
    ])

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30,
                max_iter=1000, verbose=0)
    emb = tsne.fit_transform(combined)

    # Split
    emb_bs, emb_bt = emb[:n], emb[n:2*n]
    emb_as, emb_at = emb[2*n:3*n], emb[3*n:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: before adaptation
    axes[0].scatter(emb_bs[:, 0], emb_bs[:, 1], c="#3498db", alpha=0.5,
                    s=8, label="Source")
    axes[0].scatter(emb_bt[:, 0], emb_bt[:, 1], c="#e74c3c", alpha=0.5,
                    s=8, label="Target")
    axes[0].set_title("Before Adaptation  (Source-only Model)", fontsize=13)
    axes[0].legend(markerscale=3, fontsize=11)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: after adaptation
    axes[1].scatter(emb_as[:, 0], emb_as[:, 1], c="#3498db", alpha=0.5,
                    s=8, label="Source")
    axes[1].scatter(emb_at[:, 0], emb_at[:, 1], c="#e74c3c", alpha=0.5,
                    s=8, label="Target")
    axes[1].set_title("After Adaptation  (Adversarial Model)", fontsize=13)
    axes[1].legend(markerscale=3, fontsize=11)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_training_curves(history, title, save_path, loss_key="loss"):
    """Plot loss and Dice vs epoch.

    Args:
        history: dict with training metrics
        title: plot title
        save_path: output path
        loss_key: key for the loss in history (e.g. "loss" or "seg_loss")
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history[loss_key]) + 1)

    axes[0].plot(epochs, history[loss_key], "o-", color="#3498db")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} -- Loss")
    axes[0].grid(alpha=0.3)

    dice_key = "dice"
    axes[1].plot(epochs, history[dice_key], "o-", color="#2ecc71")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title(f"{title} -- Dice")
    axes[1].grid(alpha=0.3)

    # If domain loss exists, overlay in a twin axis
    if "domain_loss" in history:
        ax_dom = axes[0].twinx()
        ax_dom.plot(epochs, history["domain_loss"], "s--", color="#e74c3c",
                    alpha=0.7, markersize=4)
        ax_dom.set_ylabel("Domain Loss", color="#e74c3c", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Experiment 6: Domain Adaptation for Segmentation")
    print(f"  Device : {DEVICE}")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 60)

    # ---- Transforms ----
    # Normalise to roughly [0, 1] then normalise with approximate mean/std
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ---- Datasets ----
    print("\n--- Generating synthetic vessel datasets ---")
    src_train = SyntheticVesselDataset(
        num_samples=NUM_SAMPLES, source_domain=True,
        transform=base_transform)
    src_test = SyntheticVesselDataset(
        num_samples=NUM_TEST_SAMPLES, source_domain=True,
        transform=base_transform)
    tgt_train = SyntheticVesselDataset(
        num_samples=NUM_SAMPLES, source_domain=False,
        transform=base_transform)
    tgt_test = SyntheticVesselDataset(
        num_samples=NUM_TEST_SAMPLES, source_domain=False,
        transform=base_transform)

    src_train_loader = DataLoader(src_train, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=NUM_WORKERS,
                                  drop_last=True)
    src_test_loader  = DataLoader(src_test,  batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS)
    tgt_train_loader = DataLoader(tgt_train, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=NUM_WORKERS,
                                  drop_last=True)
    tgt_test_loader  = DataLoader(tgt_test,  batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS)

    # Quick sanity: visualise one sample from each domain
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for row, (loader, title) in enumerate(
            [(src_train_loader, "Source Domain"),
             (tgt_train_loader, "Target Domain")]):
        batch = next(iter(loader))
        for col in range(4):
            img = batch[0][col].numpy().transpose(1, 2, 0)
            img = img * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=10)
    plt.suptitle("Sample Images (Left: Source, Right: Target)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "domain_samples.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved domain_samples.png")

    # ==================================================================
    # Phase 1: Source-only training
    # ==================================================================
    print("\n" + "-" * 55)
    print("  Phase 1: Source-only U-Net (no adaptation)")
    print("-" * 55)
    set_seed()
    model_src_only = DomainAdaptationUNet(lambda_adv=0.0).to(DEVICE)
    hist_src = train_source_only(model_src_only, src_train_loader,
                                  NUM_EPOCHS, DEVICE)
    plot_training_curves(hist_src, "Source-Only",
                         os.path.join(OUTPUT_DIR, "training_curve_source_only.png"))

    dice_src_before = evaluate(model_src_only, src_test_loader, DEVICE)
    dice_tgt_before = evaluate(model_src_only, tgt_test_loader, DEVICE)
    print(f"\n  >>> Source-only Dice on Source Test: {dice_src_before:.4f}")
    print(f"  >>> Source-only Dice on Target Test: {dice_tgt_before:.4f}")

    # ==================================================================
    # Phase 2: Adversarial training
    # ==================================================================
    print("\n" + "-" * 55)
    print("  Phase 2: Adversarial (DANN-style) training")
    print("-" * 55)
    set_seed()
    model_adv = DomainAdaptationUNet(lambda_adv=LAMBDA_ADV).to(DEVICE)
    hist_adv = train_adversarial(model_adv, src_train_loader, tgt_train_loader,
                                  NUM_EPOCHS, DEVICE, LAMBDA_ADV)
    plot_training_curves(hist_adv, "Adversarial",
                         os.path.join(OUTPUT_DIR, "training_curve_adversarial.png"),
                         loss_key="seg_loss")

    dice_src_after = evaluate(model_adv, src_test_loader, DEVICE)
    dice_tgt_after = evaluate(model_adv, tgt_test_loader, DEVICE)
    print(f"\n  >>> Adversarial Dice on Source Test: {dice_src_after:.4f}")
    print(f"  >>> Adversarial Dice on Target Test: {dice_tgt_after:.4f}")

    # ==================================================================
    # Visualisations
    # ==================================================================
    print("\n--- Visualisations ---")

    # Dice comparison (target domain)
    plot_dice_comparison(dice_tgt_before, dice_tgt_after,
                         os.path.join(OUTPUT_DIR, "dice_comparison.png"))

    # Segmentation visualisation (on target domain)
    visualize_predictions(model_src_only, model_adv, tgt_test_loader, DEVICE)

    # t-SNE of encoder features
    print("\n  Extracting features for t-SNE ...")
    feats_bs = extract_features(model_src_only, src_test_loader, DEVICE)
    feats_bt = extract_features(model_src_only, tgt_test_loader, DEVICE)
    feats_as = extract_features(model_adv,      src_test_loader, DEVICE)
    feats_at = extract_features(model_adv,      tgt_test_loader, DEVICE)

    plot_tsne(feats_bs, feats_bt, feats_as, feats_at,
              os.path.join(OUTPUT_DIR, "tsne_features.png"))

    # ==================================================================
    # Save metrics
    # ==================================================================
    results = {
        "dice_source_only_source":      float(f"{dice_src_before:.4f}"),
        "dice_source_only_target":      float(f"{dice_tgt_before:.4f}"),
        "dice_adversarial_source":      float(f"{dice_src_after:.4f}"),
        "dice_adversarial_target":      float(f"{dice_tgt_after:.4f}"),
        "dice_improvement_target":      float(f"{dice_tgt_after - dice_tgt_before:.4f}"),
        "num_epochs":                   NUM_EPOCHS,
        "batch_size":                   BATCH_SIZE,
        "learning_rate":                LR,
        "lambda_adv":                   LAMBDA_ADV,
        "num_train_samples":            NUM_SAMPLES,
    }
    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {json_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print(f"  Source Dice before adaptation:  {dice_src_before:.4f}")
    print(f"  Target Dice before adaptation:  {dice_tgt_before:.4f}")
    print(f"  Source Dice after  adaptation:  {dice_src_after:.4f}")
    print(f"  Target Dice after  adaptation:  {dice_tgt_after:.4f}")
    print(f"  Target Improvement:            +{dice_tgt_after - dice_tgt_before:.4f}")
    print(f"  Results:                        {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
