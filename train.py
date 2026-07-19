"""
train.py — Driver Drowsiness Eye Classifier (PyTorch)

Dataset: set DATA_PATH to your folder containing closed_eye/ and open_eye/
Auto-splits into 70% train / 15% val / 15% test.
Saves best model as driver_eye_model.pth and plots training curves.

Usage:
    python train.py
    python train.py --data D:/path/to/dataset --epochs 20
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import EyeCNN

# ─── ARGS ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data',   default=r'D:\Projects\DriverDrowsinessDetection',
                    help='Path to dataset folder (contains closed_eye/ open_eye/)')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch',  type=int, default=32)
parser.add_argument('--lr',     type=float, default=1e-3)
parser.add_argument('--output', default='driver_eye_model.pth')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ─────────────────────────────────────────────────────────────────


# ─── TRANSFORMS ──────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
val_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ─── HELPERS ─────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            preds  = model(imgs)
            loss   = criterion(preds, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += ((preds > 0.5).float() == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def plot_curves(history, save_path='training_curves.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'],   label='Val')
    ax1.set_title('Loss');  ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True)

    ax2.plot([a*100 for a in history['train_acc']], label='Train')
    ax2.plot([a*100 for a in history['val_acc']],   label='Val')
    ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'  📊 Training curves saved → {save_path}')


# ─── MAIN ────────────────────────────────────────────────────────
def main():
    print(f'Device : {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'Data   : {args.data}\n')

    # Dataset & splits
    full_ds  = datasets.ImageFolder(args.data, transform=train_tf)
    print(f'Classes : {full_ds.classes}')   # ['closed_eye', 'open_eye']
    print(f'Total   : {len(full_ds)} images')

    n       = len(full_ds)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    # Override val/test transforms (no augmentation)
    val_ds.dataset  = datasets.ImageFolder(args.data, transform=val_tf)
    test_ds.dataset = datasets.ImageFolder(args.data, transform=val_tf)

    print(f'Split   : train={n_train}  val={n_val}  test={n_test}\n')

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=2)

    model     = EyeCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    history          = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc     = 0.0
    patience_counter = 0
    EARLY_STOP       = 4

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f'Epoch {epoch:02d}/{args.epochs}  '
              f'train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  '
              f'val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.1f}%')

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), args.output)
            print(f'  ✅ Best model saved (val_acc={vl_acc*100:.1f}%)')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print('Early stopping triggered.')
                break

    # Plot & test
    plot_curves(history)

    model.load_state_dict(torch.load(args.output, map_location=DEVICE))
    _, test_acc = run_epoch(model, test_loader, criterion)
    print(f'\n{"─"*45}')
    print(f'  TEST ACCURACY : {test_acc*100:.2f}%')
    print(f'  Model saved   → {args.output}')
    print(f'{"─"*45}')


if __name__ == '__main__':
    main()