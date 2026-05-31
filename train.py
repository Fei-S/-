"""
Training script — Haze4K dataset.

Key improvements over v1:
    - All 3000 training images (no artificial limit)
    - L1 + SSIM combined loss (better perceptual quality)
    - 6-layer backbone with BatchNorm, mid_channels=32
    - Higher LR (1e-3) with cosine annealing
    - Gradient clipping for stability
    - No VerticalFlip (haze is horizontal-only)
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import AODNet, AODNetWithSE, SRCNN, DnCNN
from utils import AverageMeter, create_dataloaders_haze4k, plot_training_curves


# ---------------------------------------------------------------------------
# SSIM loss (differentiable)
# ---------------------------------------------------------------------------

def _gaussian_window(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w1d = g.unsqueeze(1) @ g.unsqueeze(0)  # (size, size)
    w2d = w1d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size)
    return w2d


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 - SSIM as a loss term.  Input: (B, C, H, W) in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    window = _gaussian_window(window_size, 1.5, img1.size(1), img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
    )
    return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """L1 + lambda_ssim * (1 - SSIM)"""

    def __init__(self, lambda_ssim: float = 0.15):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_l1 = self.l1(pred, target)
        loss_ssim = ssim_loss(pred, target)
        return loss_l1 + self.lambda_ssim * loss_ssim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Lightweight dehazing — Haze4K")

    # Data
    p.add_argument("--train_hazy_dir", default="./datasets/Haze4K/train/haze")
    p.add_argument("--train_gt_dir", default="./datasets/Haze4K/train/gt")
    p.add_argument("--test_hazy_dir", default="./datasets/Haze4K/test/haze")
    p.add_argument("--test_gt_dir", default="./datasets/Haze4K/test/gt")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_train", type=int, default=None, help="limit samples (None=all)")
    p.add_argument("--num_test", type=int, default=100)

    # Model
    p.add_argument("--model", default="aodnet", choices=["aodnet", "ours", "srcnn", "dncnn"])
    p.add_argument("--mid_channels", type=int, default=32)
    p.add_argument("--se_reduction", type=int, default=16)

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_ssim", type=float, default=0.15)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # System
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument("--resume", default=None)

    return p.parse_args()


def build_model(args) -> nn.Module:
    if args.model == "aodnet":
        return AODNet(in_channels=3, mid_channels=args.mid_channels)
    elif args.model == "ours":
        return AODNetWithSE(in_channels=3, mid_channels=args.mid_channels, se_reduction=args.se_reduction)
    elif args.model == "srcnn":
        return SRCNN(in_channels=3)
    elif args.model == "dncnn":
        return DnCNN(in_channels=3, depth=8, mid_channels=64)
    raise ValueError(f"Unknown model: {args.model}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total, grad_clip):
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (hazy, gt) in enumerate(loader):
        hazy, gt = hazy.to(device), gt.to(device)

        output = model(hazy)
        loss = criterion(output, gt)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_meter.update(loss.item(), hazy.size(0))

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch [{epoch}/{total}] Batch [{batch_idx+1}/{len(loader)}] Loss: {loss_meter.avg:.6f}")

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    for hazy, gt in loader:
        hazy, gt = hazy.to(device), gt.to(device)
        output = model(hazy)
        loss_meter.update(criterion(output, gt).item(), hazy.size(0))
    return loss_meter.avg


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    print("\n=== Loading Haze4K ===")
    train_loader, test_loader = create_dataloaders_haze4k(
        train_hazy_dir=args.train_hazy_dir, train_gt_dir=args.train_gt_dir,
        test_hazy_dir=args.test_hazy_dir, test_gt_dir=args.test_gt_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size, num_workers=args.num_workers,
        num_train=args.num_train, num_test=args.num_test,
    )

    # Model
    print("\n=== Building Model ===")
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()} | Params: {n_params:,} ({n_params/1e6:.3f}M)")

    # Loss & Optimizer
    criterion = CombinedLoss(lambda_ssim=args.lambda_ssim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch, best_loss = 1, float("inf")
    train_losses, val_losses = [], []
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])

    os.makedirs(args.save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{args.model}_{ts}"

    with open(os.path.join(args.save_dir, f"{save_name}_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Train
    print(f"\n=== Training {args.epochs} epochs ===")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, args.grad_clip)
        val_loss = validate(model, test_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch:3d}/{args.epochs}]  Train: {train_loss:.6f}  Val: {val_loss:.6f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}  Time: {time.time()-t_ep:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_loss": best_loss,
                "train_losses": train_losses, "val_losses": val_losses,
                "model_type": args.model, "num_params": n_params,
            }, os.path.join(args.save_dir, f"{save_name}_best.pth"))
            print(f"  [BEST] saved (val_loss={best_loss:.6f})")

        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_loss": best_loss,
                "train_losses": train_losses, "val_losses": val_losses,
            }, os.path.join(args.save_dir, f"{save_name}_epoch{epoch}.pth"))

    # Save final
    torch.save({
        "epoch": args.epochs, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "best_loss": best_loss,
        "train_losses": train_losses, "val_losses": val_losses,
    }, os.path.join(args.save_dir, f"{save_name}_final.pth"))

    elapsed = time.time() - t0
    print(f"\n=== Done ({elapsed/60:.1f} min) === Best val loss: {best_loss:.6f}")

    # Curves
    plot_training_curves(train_losses, os.path.join(args.save_dir, f"{save_name}_loss.png"),
                         title=f"{args.model.upper()} Training Loss")
    plot_training_curves(val_losses, os.path.join(args.save_dir, f"{save_name}_val_loss.png"),
                         title=f"{args.model.upper()} Validation Loss")


if __name__ == "__main__":
    main()
