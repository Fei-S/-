"""
Generate 5 comparison figures for the report.
Layout: Input | DCP | AOD-Net | Ours | SRCNN | DnCNN | GT
"""

import os, sys, time
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import AODNet, AODNetWithSE, SRCNN, DnCNN
from traditional import DCPDehazer
from utils.dataset import DehazeDataset

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
NUM_IMAGES = 5
CHECKPOINT_DIR = "./checkpoints"
TEST_HAZY = "./datasets/Haze4K/test/haze"
TEST_GT = "./datasets/Haze4K/test/gt"
OUTPUT_DIR = "./results/report_figures"

# Model paths
CKPTS = {
    "aodnet": os.path.join(CHECKPOINT_DIR, "aodnet_20260529_223401_best.pth"),
    "ours": os.path.join(CHECKPOINT_DIR, "ours_20260530_110032_best.pth"),
    "srcnn": os.path.join(CHECKPOINT_DIR, "srcnn_20260530_115138_best.pth"),
    "dncnn": os.path.join(CHECKPOINT_DIR, "dncnn_20260530_115456_best.pth"),
}


def load_model(ckpt, model_type):
    if model_type == "aodnet":
        m = AODNet(in_channels=3, mid_channels=32)
    elif model_type == "ours":
        m = AODNetWithSE(in_channels=3, mid_channels=32)
    elif model_type == "srcnn":
        m = SRCNN(in_channels=3)
    elif model_type == "dncnn":
        m = DnCNN(in_channels=3, depth=8, mid_channels=64)
    else:
        raise ValueError(model_type)
    state = torch.load(ckpt, map_location=DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    m.load_state_dict(state)
    m.to(DEVICE)
    m.eval()
    return m


@torch.no_grad()
def infer(model, img_np):
    """img_np: (H,W,3) in [0,1] → (H,W,3) in [0,1]"""
    h, w = img_np.shape[:2]
    t = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    out = model(t)
    out = out.cpu().squeeze(0).permute(1, 2, 0).numpy()
    out = np.clip(out, 0, 1)
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # Load all models
    print("Loading models...")
    models = {}
    for name, path in CKPTS.items():
        models[name] = load_model(path, name)
        p = sum(x.numel() for x in models[name].parameters() if x.requires_grad)
        print(f"  {name}: {p:,} params")

    dcp = DCPDehazer()
    print("  DCP: ready")

    # Load 5 test images
    ds = DehazeDataset(hazy_dir=TEST_HAZY, gt_dir=TEST_GT,
                       image_size=(IMAGE_SIZE, IMAGE_SIZE),
                       is_train=False, num_samples=NUM_IMAGES)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    print(f"\nGenerating {NUM_IMAGES} figures...")
    for idx, (hazy_t, gt_t) in enumerate(loader):
        hazy_np = hazy_t.squeeze(0).permute(1, 2, 0).numpy()  # (256,256,3) in [0,1]
        gt_np = gt_t.squeeze(0).permute(1, 2, 0).numpy()

        # DCP
        hazy_uint8 = (hazy_np * 255).astype(np.uint8)
        dcp_out = dcp.dehaze(hazy_uint8).astype(np.float32) / 255.0
        if dcp_out.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            dcp_out = cv2.resize(dcp_out, (IMAGE_SIZE, IMAGE_SIZE))

        # Deep models
        aod_out = infer(models["aodnet"], hazy_np)
        ours_out = infer(models["ours"], hazy_np)
        srcnn_out = infer(models["srcnn"], hazy_np)
        dncnn_out = infer(models["dncnn"], hazy_np)

        # Build 7-panel figure
        items = [
            (hazy_np, "Input"),
            (dcp_out, "DCP"),
            (aod_out, "AOD-Net"),
            (ours_out, "Ours"),
            (srcnn_out, "SRCNN"),
            (dncnn_out, "DnCNN"),
            (gt_np, "GT"),
        ]
        images = [img for img, _ in items]
        titles = [t for _, t in items]

        fig, axes = plt.subplots(1, 7, figsize=(28, 4))
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.axis("off")

        plt.tight_layout(pad=0.5)
        save_path = os.path.join(OUTPUT_DIR, f"comparison_{idx+1:02d}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [{idx+1}/{NUM_IMAGES}] {save_path}")

    print(f"\nDone! Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
