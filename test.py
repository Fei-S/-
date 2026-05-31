"""
测试脚本

功能:
    - 加载训练好的模型
    - 在测试集上计算 PSNR、SSIM
    - 生成对比可视化图
    - 输出定量结果表格
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import AODNet, AODNetWithSE, SRCNN, DnCNN
from traditional import DCPDehazer
from utils import (
    AverageMeter,
    DehazeDataset,
    calculate_psnr,
    calculate_ssim,
    make_comparison_image,
    save_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="轻量级图像去雾测试")

    # 数据 — Haze4K
    parser.add_argument(
        "--test_hazy_dir",
        type=str,
        default="./datasets/Haze4K/test/haze",
        help="测试有雾图像目录",
    )
    parser.add_argument(
        "--test_gt_dir",
        type=str,
        default="./datasets/Haze4K/test/gt",
        help="测试清晰图像目录",
    )

    # 模型
    parser.add_argument("--aod_checkpoint", default=None, help="AOD-Net checkpoint")
    parser.add_argument("--ours_checkpoint", default=None, help="Ours checkpoint")
    parser.add_argument("--srcnn_checkpoint", default=None, help="SRCNN checkpoint")
    parser.add_argument("--dncnn_checkpoint", default=None, help="DnCNN checkpoint")

    # 测试设置
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="测试设备",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="测试图像尺寸",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="结果保存目录",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=20,
        help="测试图像数量",
    )
    parser.add_argument(
        "--skip_dcp",
        action="store_true",
        help="跳过 DCP（DCP 较慢）",
    )
    parser.add_argument(
        "--mid_channels",
        type=int,
        default=16,
        help="中间特征通道数（需与训练一致）",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, model_type: str, device: str, mid_channels: int = 32):
    """Load model."""
    if model_type == "aodnet":
        model = AODNet(in_channels=3, mid_channels=mid_channels)
    elif model_type == "ours":
        model = AODNetWithSE(in_channels=3, mid_channels=mid_channels)
    elif model_type == "srcnn":
        model = SRCNN(in_channels=3)
    elif model_type == "dncnn":
        model = DnCNN(in_channels=3, depth=8, mid_channels=64)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 处理不同格式的 checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def inference_aod(model: torch.nn.Module, hazy: torch.Tensor, device: str) -> np.ndarray:
    """AOD-Net 推理"""
    model.eval()
    hazy = hazy.to(device)

    output = model(hazy)

    # 转换回 numpy，clip 到 [0,1]（SRCNN/DnCNN 无物理约束可能越界）
    output = output.cpu().squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)
    output = np.clip(output, 0.0, 1.0)
    return output


def compute_fps(model: torch.nn.Module, image_size: int, device: str, num_runs: int = 100) -> float:
    """计算推理 FPS"""
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    # 预热
    for _ in range(10):
        _ = model(dummy_input)

    # 计时
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start

    return num_runs / elapsed


def main():
    args = parse_args()

    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "comparisons"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "individual"), exist_ok=True)

    # 加载模型
    print("\n--- Load Models ---")
    models = {}  # name → (model, params)

    def _try_load(ckpt, name, model_type):
        if ckpt and os.path.exists(ckpt):
            m = load_model(ckpt, model_type, device, mid_channels=args.mid_channels)
            p = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{name}: {p:,} params ({p/1e6:.3f}M)")
            return m, p
        return None, 0

    aod_model, aod_params = _try_load(args.aod_checkpoint, "AOD-Net", "aodnet")
    ours_model, ours_params = _try_load(args.ours_checkpoint, "Ours", "ours")
    srcnn_model, srcnn_params = _try_load(args.srcnn_checkpoint, "SRCNN", "srcnn")
    dncnn_model, dncnn_params = _try_load(args.dncnn_checkpoint, "DnCNN", "dncnn")

    # DCP
    dcp_dehazer = None
    if not args.skip_dcp:
        dcp_dehazer = DCPDehazer()
        print("DCP: ready")

    # Data loading — Haze4K test set
    print("\n--- Loading Test Data (Haze4K) ---")
    test_ds = DehazeDataset(
        hazy_dir=args.test_hazy_dir,
        gt_dir=args.test_gt_dir,
        image_size=(args.image_size, args.image_size),
        is_train=False,
        num_samples=args.num_images,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
    )

    # 限制测试数量
    total_images = min(len(test_loader.dataset), args.num_images)
    print(f"测试图像数量: {total_images}")

    # 指标统计
    dcp_psnr_meter = AverageMeter()
    dcp_ssim_meter = AverageMeter()
    aod_psnr_meter = AverageMeter()
    aod_ssim_meter = AverageMeter()
    ours_psnr_meter = AverageMeter()
    ours_ssim_meter = AverageMeter()
    srcnn_psnr_meter = AverageMeter()
    srcnn_ssim_meter = AverageMeter()
    dncnn_psnr_meter = AverageMeter()
    dncnn_ssim_meter = AverageMeter()

    dcp_time_meter = AverageMeter()
    aod_time_meter = AverageMeter()
    ours_time_meter = AverageMeter()
    srcnn_time_meter = AverageMeter()
    dncnn_time_meter = AverageMeter()

    # --- 逐张测试 ---
    print("\n--- 开始测试 ---")
    for idx, (hazy_tensor, gt_tensor) in enumerate(tqdm(test_loader, total=total_images)):
        if idx >= total_images:
            break

        # 转换为 numpy 便于处理
        gt_np = gt_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)
        hazy_np = hazy_tensor.squeeze(0).permute(1, 2, 0).numpy()

        # --- DCP ---
        dcp_output = None
        if dcp_dehazer is not None:
            try:
                hazy_uint8 = (hazy_np * 255).astype(np.uint8)
                t_start = time.time()
                dcp_output = dcp_dehazer.dehaze(hazy_uint8)
                dcp_time = time.time() - t_start

                dcp_output_float = dcp_output.astype(np.float64) / 255.0

                # 需要 resize 到和 gt 一致（DCP 保持原始尺寸）
                if dcp_output_float.shape[:2] != gt_np.shape[:2]:
                    from cv2 import resize as cv2_resize

                    dcp_output_float = cv2_resize(
                        dcp_output_float,
                        (gt_np.shape[1], gt_np.shape[0]),
                    )

                dcp_psnr = calculate_psnr(dcp_output_float, gt_np)
                dcp_ssim = calculate_ssim(dcp_output_float, gt_np)
                dcp_psnr_meter.update(dcp_psnr)
                dcp_ssim_meter.update(dcp_ssim)
                dcp_time_meter.update(dcp_time)
            except Exception as e:
                print(f"\n[警告] DCP 处理失败 (图像 {idx}): {e}")
                dcp_output = None

        # --- AOD-Net ---
        t_start = time.time()
        aod_output = inference_aod(aod_model, hazy_tensor, device)
        aod_time = time.time() - t_start

        aod_psnr = calculate_psnr(aod_output, gt_np)
        aod_ssim = calculate_ssim(aod_output, gt_np)
        aod_psnr_meter.update(aod_psnr)
        aod_ssim_meter.update(aod_ssim)
        aod_time_meter.update(aod_time)

        # --- Ours ---
        ours_output = None
        if ours_model is not None:
            t_start = time.time()
            ours_output = inference_aod(ours_model, hazy_tensor, device)
            ours_time = time.time() - t_start

            ours_psnr = calculate_psnr(ours_output, gt_np)
            ours_ssim = calculate_ssim(ours_output, gt_np)
            ours_psnr_meter.update(ours_psnr)
            ours_ssim_meter.update(ours_ssim)
            ours_time_meter.update(ours_time)

        # --- SRCNN ---
        srcnn_output = None
        if srcnn_model is not None:
            t_start = time.time()
            srcnn_output = inference_aod(srcnn_model, hazy_tensor, device)
            srcnn_time = time.time() - t_start

            srcnn_psnr = calculate_psnr(srcnn_output, gt_np)
            srcnn_ssim = calculate_ssim(srcnn_output, gt_np)
            srcnn_psnr_meter.update(srcnn_psnr)
            srcnn_ssim_meter.update(srcnn_ssim)
            srcnn_time_meter.update(srcnn_time)

        # --- DnCNN ---
        dncnn_output = None
        if dncnn_model is not None:
            t_start = time.time()
            dncnn_output = inference_aod(dncnn_model, hazy_tensor, device)
            dncnn_time = time.time() - t_start

            dncnn_psnr = calculate_psnr(dncnn_output, gt_np)
            dncnn_ssim = calculate_ssim(dncnn_output, gt_np)
            dncnn_psnr_meter.update(dncnn_psnr)
            dncnn_ssim_meter.update(dncnn_ssim)
            dncnn_time_meter.update(dncnn_time)

        # --- 保存对比图 ---
        comparison_path = os.path.join(
            args.save_dir, "comparisons", f"comparison_{idx:03d}.png"
        )
        make_comparison_image(
            hazy=hazy_np,
            dcp_output=dcp_output.astype(np.float64) / 255.0 if dcp_output is not None else None,
            aod_output=aod_output,
            ours_output=ours_output,
            gt=gt_np,
            save_path=comparison_path,
        )

        # --- 保存各方法单独结果 ---
        if aod_output is not None:
            save_image(
                (aod_output * 255).astype(np.uint8) if aod_output.max() <= 1 else aod_output,
                os.path.join(args.save_dir, "individual", f"aod_{idx:03d}.png"),
            )
        if ours_output is not None:
            save_image(
                (ours_output * 255).astype(np.uint8) if ours_output.max() <= 1 else ours_output,
                os.path.join(args.save_dir, "individual", f"ours_{idx:03d}.png"),
            )

    # --- FPS 测试 ---
    print("\n--- 计算 FPS ---")
    aod_fps = compute_fps(aod_model, args.image_size, device)
    print(f"AOD-Net FPS: {aod_fps:.2f}")

    ours_fps = 0
    if ours_model is not None:
        ours_fps = compute_fps(ours_model, args.image_size, device)
        print(f"Ours FPS:    {ours_fps:.2f}")

    srcnn_fps = 0
    if srcnn_model is not None:
        srcnn_fps = compute_fps(srcnn_model, args.image_size, device)
        print(f"SRCNN FPS:   {srcnn_fps:.2f}")

    dncnn_fps = 0
    if dncnn_model is not None:
        dncnn_fps = compute_fps(dncnn_model, args.image_size, device)
        print(f"DnCNN FPS:   {dncnn_fps:.2f}")

    # --- 输出结果表格 ---
    print("\n" + "=" * 65)
    print("                    定量结果对比")
    print("=" * 65)
    print(f"{'Method':<15} {'Params':<12} {'PSNR(dB)':<12} {'SSIM':<12} {'FPS':<12}")
    print("-" * 65)

    if dcp_dehazer is not None:
        print(
            f"{'DCP':<15} {'-':<12} "
            f"{dcp_psnr_meter.avg:<12.2f} {dcp_ssim_meter.avg:<12.4f} "
            f"{1.0 / dcp_time_meter.avg if dcp_time_meter.avg > 0 else 0:<12.2f}"
        )

    print(
        f"{'AOD-Net':<15} {f'{aod_params/1e6:.3f}M':<12} "
        f"{aod_psnr_meter.avg:<12.2f} {aod_ssim_meter.avg:<12.4f} "
        f"{aod_fps:<12.2f}"
    )

    if ours_model is not None:
        print(
            f"{'Ours':<15} {f'{ours_params/1e6:.3f}M':<12} "
            f"{ours_psnr_meter.avg:<12.2f} {ours_ssim_meter.avg:<12.4f} "
            f"{ours_fps:<12.2f}"
        )
    if srcnn_model is not None:
        print(
            f"{'SRCNN':<15} {f'{srcnn_params/1e6:.3f}M':<12} "
            f"{srcnn_psnr_meter.avg:<12.2f} {srcnn_ssim_meter.avg:<12.4f} "
            f"{srcnn_fps:<12.2f}"
        )
    if dncnn_model is not None:
        print(
            f"{'DnCNN':<15} {f'{dncnn_params/1e6:.3f}M':<12} "
            f"{dncnn_psnr_meter.avg:<12.2f} {dncnn_ssim_meter.avg:<12.4f} "
            f"{dncnn_fps:<12.2f}"
        )

    print("=" * 65)

    # --- 保存结果到文件 ---
    results_path = os.path.join(args.save_dir, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("轻量级图像去雾实验结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"测试图像数量: {total_images}\n")
        f.write(f"图像尺寸: {args.image_size}x{args.image_size}\n\n")
        f.write(f"{'Method':<15} {'Params':<12} {'PSNR(dB)':<12} {'SSIM':<12} {'FPS':<12}\n")
        f.write("-" * 50 + "\n")

        if dcp_dehazer is not None:
            f.write(
                f"{'DCP':<15} {'-':<12} "
                f"{dcp_psnr_meter.avg:<12.2f} {dcp_ssim_meter.avg:<12.4f} "
                f"{1.0 / dcp_time_meter.avg if dcp_time_meter.avg > 0 else 0:<12.2f}\n"
            )

        f.write(
            f"{'AOD-Net':<15} {f'{aod_params/1e6:.3f}M':<12} "
            f"{aod_psnr_meter.avg:<12.2f} {aod_ssim_meter.avg:<12.4f} "
            f"{aod_fps:<12.2f}\n"
        )

        if ours_model is not None:
            f.write(
                f"{'Ours':<15} {f'{ours_params/1e6:.3f}M':<12} "
                f"{ours_psnr_meter.avg:<12.2f} {ours_ssim_meter.avg:<12.4f} "
                f"{ours_fps:<12.2f}\n"
            )
        if srcnn_model is not None:
            f.write(
                f"{'SRCNN':<15} {f'{srcnn_params/1e6:.3f}M':<12} "
                f"{srcnn_psnr_meter.avg:<12.2f} {srcnn_ssim_meter.avg:<12.4f} "
                f"{srcnn_fps:<12.2f}\n"
            )
        if dncnn_model is not None:
            f.write(
                f"{'DnCNN':<15} {f'{dncnn_params/1e6:.3f}M':<12} "
                f"{dncnn_psnr_meter.avg:<12.2f} {dncnn_ssim_meter.avg:<12.4f} "
                f"{dncnn_fps:<12.2f}\n"
            )

    print(f"\n结果已保存至: {results_path}")
    print(f"对比图已保存至: {os.path.join(args.save_dir, 'comparisons/')}")


if __name__ == "__main__":
    main()
