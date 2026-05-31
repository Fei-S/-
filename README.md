# 轻量级恶劣天气图像复原

基于轻量注意力机制的单图像去雾方法 — 《高级机器学习理论》课程项目。

## 项目概述

实现三种去雾方法并在 RTX2060 本地运行：

| 方法 | 类型 | 说明 |
|---|---|---|
| DCP | 传统方法 | Dark Channel Prior，无需训练，经典 baseline |
| AOD-Net | 轻量深度学习 | 超轻量 CNN，5层卷积 |
| Ours | AOD-Net + SE Attention | 引入通道注意力增强特征表达 |

## 环境配置

```bash
# 创建虚拟环境
conda create -n dehaze python=3.10
conda activate dehaze

# 安装依赖
pip install -r requirements.txt
```

## 项目结构

```
dehaze_project/
├── models/
│   ├── aodnet.py       # AOD-Net 网络
│   ├── se_block.py     # SE Attention 模块
│   └── ours_model.py    # AOD-Net + SE
├── traditional/
│   └── dcp.py           # DCP 传统方法
├── utils/
│   ├── dataset.py       # 数据加载
│   ├── metrics.py       # PSNR / SSIM 指标
│   └── visualize.py     # 结果可视化
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── inference.py          # 单张推理
├── datasets/            # 数据集目录
│   ├── train/           # 有雾训练图像
│   ├── test/            # 有雾测试图像
│   └── gt/              # 清晰真值图像
├── results/             # 结果输出
└── checkpoints/         # 模型权重
```

## 数据集准备

使用 RESIDE SOTS 数据集（建议仅使用 100 张训练 + 20 张测试）：

```bash
# 将数据按以下结构放置：
datasets/
├── train/    # 有雾训练图像 (100张)
├── test/     # 有雾测试图像 (20张)
└── gt/       # 清晰真值图像 (对应)
```

## 使用方法

### 1. 训练 AOD-Net

```bash
python train.py \
    --model aodnet \
    --hazy_dir ./datasets/train \
    --gt_dir ./datasets/gt \
    --batch_size 4 \
    --epochs 30 \
    --lr 1e-4
```

### 2. 训练 Ours (AOD-Net + SE)

```bash
python train.py \
    --model ours \
    --hazy_dir ./datasets/train \
    --gt_dir ./datasets/gt \
    --batch_size 4 \
    --epochs 30 \
    --lr 1e-4
```

### 3. 测试与评估

```bash
python test.py \
    --hazy_dir ./datasets/test \
    --gt_dir ./datasets/gt \
    --aod_checkpoint ./checkpoints/aodnet_best.pth \
    --ours_checkpoint ./checkpoints/ours_best.pth \
    --num_images 20
```

### 4. 单张图像推理

```bash
python inference.py \
    --input ./test_image.jpg \
    --aod_checkpoint ./checkpoints/aodnet_best.pth \
    --ours_checkpoint ./checkpoints/ours_best.pth \
    --output_dir ./results/inference
```

## 实验结果

| Method | Params | PSNR (dB) | SSIM | FPS |
|---|---|---|---|---|
| DCP | - | xx | xx | xx |
| AOD-Net | 0.007M | xx | xx | xx |
| Ours | 0.007M | xx | xx | xx |

*具体数值需运行后填入。*

## 技术要点

- **AOD-Net**: 通过 K(x) 变换映射实现端到端去雾
- **SE Attention**: Squeeze-and-Excitation 通道注意力，增强雾区域感知
- **轻量化**: 模型参数约 7K，RTX2060 可轻松训练
- **推理速度**: RTX2060 上可达 100+ FPS

## License

仅用于课程学习。
