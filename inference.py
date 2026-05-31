"""
单张图像推理脚本

功能:
    - 对单张图像使用所有方法去雾
    - 自动生成对比图
    - 支持批量处理
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from models import AODNet, AODNetWithSE
from traditional import DCPDehazer
from utils import make_comparison_image, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="单张图像去雾推理")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入图像路径（单张或目录）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/inference",
        help="输出目录",
    )

    # 模型路径
    parser.add_argument(
        "--aod_checkpoint",
        type=str,
        default=None,
        help="AOD-Net 模型权重路径",
    )
    parser.add_argument(
        "--ours_checkpoint",
        type=str,
        default=None,
        help="Ours 模型权重路径",
    )

    # 设置
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    parser.add_argument(
        "--skip_dcp",
        action="store_true",
        help="跳过 DCP（加快速度）",
    )
    parser.add_argument(
        "--mid_channels",
        type=int,
        default=16,
        help="中间特征通道数",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="推理前 resize 到的尺寸（短边）",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, model_type: str, device: str, mid_channels: int = 16):
    """加载模型"""
    if checkpoint_path is None:
        return None

    if model_type == "aodnet":
        model = AODNet(in_channels=3, mid_channels=mid_channels)
    elif model_type == "ours":
        model = AODNetWithSE(in_channels=3, mid_channels=mid_channels)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def inference_deep_model(
    model: torch.nn.Module,
    img_np: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    使用深度学习模型推理

    Args:
        model: 神经网络模型
        img_np: 输入图像 (H, W, 3)，值域 [0, 1]
        device: 设备

    Returns:
        去雾图像 (H, W, 3)，值域 [0, 1]
    """
    h, w = img_np.shape[:2]

    # 转换为 tensor
    img_tensor = (
        torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0)
    )  # (1, 3, H, W)
    img_tensor = img_tensor.to(device)

    # 推理
    output = model(img_tensor)

    # 转回 numpy
    output_np = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
    output_np = np.clip(output_np, 0, 1)

    # 如果尺寸变了，resize 回去
    if output_np.shape[:2] != (h, w):
        output_np = cv2.resize(
            output_np, (w, h), interpolation=cv2.INTER_LINEAR
        )

    return output_np


def process_single_image(
    image_path: str,
    dcp_dehazer: DCPDehazer | None,
    aod_model: torch.nn.Module | None,
    ours_model: torch.nn.Module | None,
    output_dir: str,
    device: str,
    resize: int | None = None,
):
    """
    处理单张图像

    Args:
        image_path: 图像路径
        dcp_dehazer: DCP 去雾器
        aod_model: AOD-Net 模型
        ours_model: Ours 模型
        output_dir: 输出目录
        device: 设备
        resize: resize 尺寸
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"[错误] 无法读取图像: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    original_size = img_rgb.shape[:2]
    if resize is not None:
        h, w = img_rgb.shape[:2]
        if h < w:
            new_h = resize
            new_w = int(w * resize / h)
        else:
            new_w = resize
            new_h = int(h * resize / w)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    img_name = Path(image_path).stem

    # 转为 [0, 1] 范围供深度学习模型使用
    img_float = img_rgb.astype(np.float32) / 255.0

    # --- DCP ---
    dcp_output = None
    if dcp_dehazer is not None:
        print(f"  [DCP] processing...")
        dcp_output = dcp_dehazer.dehaze(img_rgb)
        save_image(
            dcp_output,
            os.path.join(output_dir, f"{img_name}_dcp.png"),
            convert_bgr=True,
        )
        # 转为 [0, 1]
        dcp_output = dcp_output.astype(np.float32) / 255.0

    # --- AOD-Net ---
    aod_output = None
    if aod_model is not None:
        print(f"  [AODNet] processing...")
        aod_output = inference_deep_model(aod_model, img_float, device)
        save_image(
            (aod_output * 255).astype(np.uint8),
            os.path.join(output_dir, f"{img_name}_aod.png"),
        )

    # --- Ours ---
    ours_output = None
    if ours_model is not None:
        print(f"  [Ours] processing...")
        ours_output = inference_deep_model(ours_model, img_float, device)
        save_image(
            (ours_output * 255).astype(np.uint8),
            os.path.join(output_dir, f"{img_name}_ours.png"),
        )

    # --- 生成对比图 ---
    print(f"  [Viz] generating comparison...")
    comparison_path = os.path.join(output_dir, f"{img_name}_comparison.png")
    make_comparison_image(
        hazy=img_float,
        dcp_output=dcp_output,
        aod_output=aod_output,
        ours_output=ours_output,
        gt=None,  # 推理时没有 GT
        save_path=comparison_path,
        titles=["Input", "DCP", "AOD-Net", "Ours", "GT (N/A)"],
    )

    print(f"  [OK] done")


def main():
    args = parse_args()

    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("--- 加载模型 ---")
    aod_model = load_model(args.aod_checkpoint, "aodnet", device, args.mid_channels)
    ours_model = load_model(args.ours_checkpoint, "ours", device, args.mid_channels)

    if aod_model is not None:
        num_params = sum(p.numel() for p in aod_model.parameters() if p.requires_grad)
        print(f"AOD-Net:   {num_params:,} 参数")
    if ours_model is not None:
        num_params = sum(p.numel() for p in ours_model.parameters() if p.requires_grad)
        print(f"Ours:      {num_params:,} 参数")

    # DCP
    dcp_dehazer = None if args.skip_dcp else DCPDehazer()
    if dcp_dehazer:
        print("DCP:  已就绪")

    # 收集输入图像
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_paths = sorted(
            [
                str(input_path / f)
                for f in os.listdir(input_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]
        )
    else:
        print(f"[错误] 输入路径不存在: {args.input}")
        return

    print(f"\n--- 处理 {len(image_paths)} 张图像 ---")
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i + 1}/{len(image_paths)}] {Path(img_path).name}")
        process_single_image(
            image_path=img_path,
            dcp_dehazer=dcp_dehazer,
            aod_model=aod_model,
            ours_model=ours_model,
            output_dir=args.output_dir,
            device=device,
            resize=args.resize,
        )

    print(f"\n--- 全部完成 ---")
    print(f"结果保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
