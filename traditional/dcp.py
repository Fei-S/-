"""
Dark Channel Prior (DCP) — 传统单图像去雾方法

论文: Single Image Haze Removal Using Dark Channel Prior (CVPR 2009, TPAMI 2011)
作者: Kaiming He, Jian Sun, Xiaoou Tang

算法步骤:
    1. 计算暗通道 (Dark Channel)
    2. 估计大气光 A (Atmospheric Light)
    3. 估计透射率 t(x) (Transmission Map)
    4. 导向滤波细化透射率 (Guided Filter)
    5. 恢复无雾图像 J(x) = (I(x) - A) / max(t(x), t0) + A
"""

import cv2
import numpy as np


class DCPDehazer:
    """
    Dark Channel Prior 去雾器

    参数:
        omega:     去雾强度系数，默认 0.95（保留少量雾感，更自然）
        t0:        透射率下限，默认 0.1（防止除零）
        patch_size: 暗通道计算窗口大小，默认 15
        guide_radius: 导向滤波半径，默认 40
        guide_eps:   导向滤波正则化参数，默认 1e-6
    """

    def __init__(
        self,
        omega: float = 0.95,
        t0: float = 0.1,
        patch_size: int = 15,
        guide_radius: int = 40,
        guide_eps: float = 1e-6,
    ):
        self.omega = omega
        self.t0 = t0
        self.patch_size = patch_size
        self.guide_radius = guide_radius
        self.guide_eps = guide_eps

    def _dark_channel(self, img: np.ndarray) -> np.ndarray:
        """
        计算暗通道图像

        暗通道定义:
            J_dark(x) = min_{y∈Ω(x)} ( min_{c∈{r,g,b}} J_c(y) )

        Args:
            img: 输入图像 (H, W, 3)，值域 [0, 255]

        Returns:
            暗通道图像 (H, W)
        """
        # 逐通道最小值
        min_channel = np.min(img, axis=2)  # (H, W)

        # 局部最小值滤波
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        dark = cv2.erode(min_channel, kernel)

        return dark

    def _estimate_atmospheric_light(
        self, img: np.ndarray, dark_channel: np.ndarray
    ) -> np.ndarray:
        """
        估计大气光 A

        步骤:
            1. 选取暗通道中最亮的 0.1% 像素
            2. 在这些像素中选取原图最亮的像素值作为大气光

        Args:
            img:          输入图像 (H, W, 3)，值域 [0, 255]
            dark_channel: 暗通道 (H, W)

        Returns:
            大气光 A (3,) — RGB 三通道值
        """
        h, w = dark_channel.shape
        num_pixels = h * w
        num_top = max(int(num_pixels * 0.001), 1)  # 前 0.1%

        # 找出暗通道中最亮的像素位置
        flat_dark = dark_channel.ravel()
        indices = np.argpartition(flat_dark, -num_top)[-num_top:]
        indices = indices[np.argsort(flat_dark[indices])[::-1]]

        # 在原图中取出这些位置的像素
        flat_img = img.reshape(-1, 3)
        top_pixels = flat_img[indices]  # (num_top, 3)

        # 以这些像素中最大的值作为大气光
        A = np.max(top_pixels, axis=0).astype(np.float64)

        return A

    def _estimate_transmission(
        self, img: np.ndarray, A: np.ndarray
    ) -> np.ndarray:
        """
        估计透射率 t(x)

        公式:
            t(x) = 1 - omega * min_{y∈Ω(x)} ( min_c (I_c(y) / A_c) )

        Args:
            img: 输入图像 (H, W, 3)，值域 [0, 255]
            A:   大气光 (3,)

        Returns:
            透射率图 t(x) (H, W)，值域 [0, 1]
        """
        # 归一化: I / A
        normalized = img.astype(np.float64) / A  # (H, W, 3)
        normalized = np.clip(normalized, 0, 1)

        # 逐通道取最小值
        min_channel = np.min(normalized, axis=2)  # (H, W)

        # 局部最小值滤波
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        dark_normalized = cv2.erode(min_channel, kernel)

        # 透射率
        transmission = 1.0 - self.omega * dark_normalized

        return np.clip(transmission, 0, 1).astype(np.float32)

    def _guided_filter(
        self, guide: np.ndarray, src: np.ndarray
    ) -> np.ndarray:
        """
        导向滤波 — 细化透射率

        Args:
            guide: 引导图像 (H, W) 或 (H, W, C)，值域 [0, 255]
            src:   待滤波图像 (H, W)，float32

        Returns:
            滤波后图像 (H, W)，float32
        """
        # 确保 guide 是灰度图
        if guide.ndim == 3:
            guide_gray = cv2.cvtColor(guide.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            guide_gray = guide.astype(np.uint8)

        # OpenCV 导向滤波
        refined = cv2.ximgproc.guidedFilter(
            guide=guide_gray,
            src=src,
            radius=self.guide_radius,
            eps=self.guide_eps,
            dDepth=-1,
        )

        return refined

    def _recover(
        self, img: np.ndarray, A: np.ndarray, transmission: np.ndarray
    ) -> np.ndarray:
        """
        恢复无雾图像

        公式:
            J(x) = (I(x) - A) / max(t(x), t0) + A

        Args:
            img:          输入图像 (H, W, 3)，值域 [0, 255]
            A:            大气光 (3,)
            transmission: 透射率图 (H, W)，值域 [0, 1]

        Returns:
            去雾图像 (H, W, 3)，值域 [0, 255]
        """
        transmission = np.maximum(transmission, self.t0)
        transmission = np.expand_dims(transmission, axis=2)  # (H, W, 1)

        # 恢复
        img_float = img.astype(np.float64)
        A_expanded = A.reshape(1, 1, 3)

        J = (img_float - A_expanded) / transmission + A_expanded

        J = np.clip(J, 0, 255).astype(np.uint8)

        return J

    def dehaze(self, img: np.ndarray) -> np.ndarray:
        """
        执行去雾

        Args:
            img: 有雾图像 (H, W, 3)，值域 [0, 255]，uint8

        Returns:
            去雾图像 (H, W, 3)，值域 [0, 255]，uint8
        """
        # 1. 计算暗通道
        dark = self._dark_channel(img)

        # 2. 估计大气光
        A = self._estimate_atmospheric_light(img, dark)

        # 3. 估计透射率
        t = self._estimate_transmission(img, A)

        # 4. 导向滤波细化透射率
        try:
            t_refined = self._guided_filter(img, t)
        except (cv2.error, AttributeError):
            # 如果 OpenCV 版本不支持 ximgproc，使用原始透射率
            print("[DCP] 警告: 导向滤波不可用，使用原始透射率")
            t_refined = t

        # 5. 恢复无雾图像
        result = self._recover(img, A, t_refined)

        return result


def dehaze_image(image_path: str, output_path: str | None = None) -> np.ndarray:
    """
    对单张图像执行 DCP 去雾

    Args:
        image_path:  输入图像路径
        output_path: 输出图像路径（可选）

    Returns:
        去雾图像 (H, W, 3) uint8
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 去雾
    dehazer = DCPDehazer()
    result = dehazer.dehaze(img_rgb)

    # 保存
    if output_path is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"结果已保存: {output_path}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python dcp.py <input_image> [output_image]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    dehaze_image(input_path, output_path)
