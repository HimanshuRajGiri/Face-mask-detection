"""
Face Mask Detection - CNN Training Script (PyTorch)
====================================================
Trains a custom CNN on the Face Mask Dataset.
Saves model as: mask_cnn_model.pth
"""

import os
import sys
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATASET_TEST_DIR   = Path("Face Mask Dataset/Test")
DATASET_TRAIN_DIR  = Path("Face Mask Dataset/Train")
IMG_SIZE           = 64      # faster on CPU
BATCH_SIZE         = 64      # larger batch = fewer iterations
EPOCHS             = 15      # early stop handles the rest
LR                 = 1e-3
PATIENCE           = 5
SEED               = 42
MODEL_OUT          = "mask_cnn_model.pth"
HISTORY_OUT        = "training_history.png"

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class MaskDataset(Dataset):
    """Loads images from WithMask / WithoutMask folders."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_image_paths(base_dirs):
    """
    Collect (path, label) pairs from directories.
    label: 1 = WithMask, 0 = WithoutMask
    """
    paths, labels = [], []
    for base in base_dirs:
        base = Path(base)
        for cls, label in [("WithMask", 1), ("WithoutMask", 0)]:
            folder = base / cls
            if not folder.exists():
                print(f"[WARN] Folder not found: {folder}")
                continue
            imgs = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
            # skip tiny corrupt files
            imgs = [p for p in imgs if p.stat().st_size > 500]
            paths.extend(imgs)
            labels.extend([label] * len(imgs))
            print(f"[INFO] {folder.name}/{cls}: {len(imgs)} images")
    return paths, labels


# ─────────────────────────────────────────────
# CNN Architecture
# ─────────────────────────────────────────────
class MaskCNN(nn.Module):
    """
    Lightweight custom CNN for binary mask classification.
    Input: 3 x 128 x 128
    Output: sigmoid probability (1 = mask, 0 = no mask)
    """

    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
            )

        self.features = nn.Sequential(
            conv_block(3,   32),   # → 32x32
            conv_block(32,  64),   # → 16x16
            conv_block(64, 128),   # → 8x8
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # → 128x1x1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x.squeeze(1)


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct    += ((preds > 0.5).float() == labels).sum().item()
        total      += len(imgs)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            preds = model(imgs)
            loss  = criterion(preds, labels)
            total_loss += loss.item() * len(imgs)
            correct    += ((preds > 0.5).float() == labels).sum().item()
            total      += len(imgs)
    return total_loss / total, correct / total


def plot_history(train_acc, val_acc, train_loss, val_loss, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training History", fontsize=16, fontweight='bold')

    ax1.plot(train_acc, label='Train Acc', color='#4CAF50', linewidth=2)
    ax1.plot(val_acc,   label='Val Acc',   color='#2196F3', linewidth=2)
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(train_loss, label='Train Loss', color='#FF5722', linewidth=2)
    ax2.plot(val_loss,   label='Val Loss',   color='#9C27B0', linewidth=2)
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Training history saved → {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # --- Load all image paths ---
    search_dirs = [DATASET_TEST_DIR, DATASET_TRAIN_DIR]
    all_paths, all_labels = load_image_paths(search_dirs)

    if len(all_paths) == 0:
        print("[ERROR] No images found! Check dataset paths.")
        sys.exit(1)

    mask_count   = sum(all_labels)
    nomask_count = len(all_labels) - mask_count
    print(f"\n[INFO] Total images: {len(all_paths)}")
    print(f"[INFO]   WithMask:    {mask_count}")
    print(f"[INFO]   WithoutMask: {nomask_count}")

    # --- Transforms ---
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # --- Train / Val / Test split (70/15/15) ---
    n      = len(all_paths)
    n_test = int(0.15 * n)
    n_val  = int(0.15 * n)
    n_train = n - n_val - n_test

    # Create full dataset with train transform first (we'll override for val/test)
    full_ds = MaskDataset(all_paths, all_labels, transform=train_tf)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    # Override transforms for val/test
    val_ds.dataset  = MaskDataset(all_paths, all_labels, transform=val_tf)
    test_ds.dataset = MaskDataset(all_paths, all_labels, transform=val_tf)

    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- Weighted sampler to handle class imbalance ---
    train_labels = [all_labels[i] for i in train_ds.indices]
    n_pos = sum(train_labels)           # WithMask
    n_neg = len(train_labels) - n_pos   # WithoutMask
    print(f"[INFO] Train class dist — WithMask: {n_pos} | WithoutMask: {n_neg}")

    w_pos = 1.0 / max(n_pos, 1)
    w_neg = 1.0 / max(n_neg, 1)
    sample_weights = [w_pos if lbl == 1 else w_neg for lbl in train_labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # --- Model ---
    model = MaskCNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[INFO] Model parameters: {total_params:,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # --- Training loop ---
    best_val_loss = float('inf')
    best_val_acc  = 0.0
    patience_cnt  = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    print("\n" + "="*60)
    print("  Starting CNN Training")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc = eval_epoch(model, val_loader,   criterion, DEVICE)
        elapsed = time.time() - t0

        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)

        scheduler.step(vl_loss)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}% | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_val_acc  = vl_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': vl_acc,
                'val_loss': vl_loss,
            }, MODEL_OUT)
            print(f"  [SAVED] Best model (val_acc={vl_acc*100:.2f}%)")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"{'='*60}\n")

    # --- Test evaluation ---
    # Load best model
    checkpoint = torch.load(MODEL_OUT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"[INFO] Test Accuracy: {test_acc*100:.2f}%")
    print(f"[INFO] Test Loss:     {test_loss:.4f}")
    print(f"[INFO] Model saved → {MODEL_OUT}")

    # --- Plot history ---
    plot_history(
        history['train_acc'], history['val_acc'],
        history['train_loss'], history['val_loss'],
        HISTORY_OUT
    )


if __name__ == "__main__":
    main()
