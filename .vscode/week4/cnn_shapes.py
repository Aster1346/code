"""
Week 4 Part 5: Image Tensor 与 CNN 的 Shape / 参数量

覆盖本周实验要求 3：
- Image Tensor 的 [N, C, H, W] 布局；
- Conv2d 的输入输出 Shape、输出尺寸公式、参数量公式；
- 卷积 vs 全连接的参数量对比（参数共享的意义）；
- 用 sklearn digits 的 8x8 图像可视化 Feature Map；
- Receptive Field（感受野）的计算。

不训练完整 CNN，只观察结构与 Shape。

运行：python 07_cnn_shapes.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 见文档 Part 11 排错表

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------- 1. Image Tensor ----------
line("1. Image Tensor: [N, C, H, W]")
images = torch.randn(2, 3, 32, 32)
print(f"一个 batch 的输入: {images.shape}")
print(f"  N (batch size) = {images.shape[0]}")
print(f"  C (channels)   = {images.shape[1]}   <- RGB 三通道")
print(f"  H x W          = {images.shape[2]} x {images.shape[3]}")
print(f"  单张图像元素个数 = {images[0].numel()}  (3*32*32)")
print("\n注意：PyTorch 用 channel-first (N,C,H,W)；")
print("      matplotlib/PIL 读进来是 HWC，输入网络前常用 .permute(2,0,1)。")

# ---------- 2. Conv2d 的 Shape 追踪 ----------
line("2. 一层 Conv + ReLU + MaxPool 的 Shape 变化")
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)

with torch.no_grad():
    features = conv(images)
print(f"输入 images           : {images.shape}")
print(f"conv(3 -> 8, k=3, p=1): {features.shape}    <- 输出通道变成 8，长宽不变")
act = torch.relu(features)
print(f"relu(features)        : {act.shape}    <- 激活不改变 shape")
pooled = pool(act)
print(f"maxpool(2x2)          : {pooled.shape}    <- 长宽各减半")

# ---------- 3. 输出尺寸公式 ----------
line("3. 输出尺寸公式：H_out = floor((H + 2p - k) / s) + 1")

def out_size(h, k, p, s):
    return (h + 2 * p - k) // s + 1

configs = [
    (32, 3, 1, 1, "本周示例"),
    (32, 3, 0, 1, "valid padding，边界被丢掉"),
    (32, 3, 0, 2, "stride=2，下采样"),
    (32, 5, 2, 1, "kernel=5，padding=2 保持尺寸"),
    (8, 3, 1, 1, "digits 8x8 图像"),
]
print(f"{'H_in':>5}{'k':>4}{'p':>4}{'s':>4}{'H_out':>7}   说明")
print("-" * 70)
for h, k, p, s, note in configs:
    print(f"{h:>5}{k:>4}{p:>4}{s:>4}{out_size(h, k, p, s):>7}   {note}")

# 用 PyTorch 实测核对
x_check = torch.randn(1, 1, 32, 32)
for k, p, s in [(3, 0, 1), (3, 0, 2), (5, 2, 1)]:
    y = nn.Conv2d(1, 1, kernel_size=k, padding=p, stride=s)(x_check)
    assert y.shape[-1] == out_size(32, k, p, s), "公式与实测不一致"
print("\n公式与 PyTorch 实测结果一致 ✓")

# ---------- 4. 参数量 ----------
line("4. 参数量：Conv 与 Linear 的对比")
conv_params = conv.weight.numel() + conv.bias.numel()
print(f"Conv2d(3, 8, k=3): weight {tuple(conv.weight.shape)} = {conv.weight.numel()}, "
      f"bias = {conv.bias.numel()}, 合计 {conv_params}")
print(f"  公式: C_out * (C_in * k * k + 1) = 8 * (3*9 + 1) = {8 * (3 * 9 + 1)}")
print("  关键点：参数量只取决于 C_out, C_in, k，与输入图像的长宽无关；")
print("          同一个 kernel 在所有位置复用 = Parameter Sharing。")

dense_in = 3 * 32 * 32
dense_out = 512
dense_params = dense_in * dense_out + dense_out
print(f"\n对照：全连接 {dense_in} -> {dense_out} 需要 {dense_params:,} 个参数")
print(f"      同样输入下 3->8 的 3x3 卷积只要 {conv_params} 个参数，"
      f"是它的 1/{dense_params // conv_params} 左右")
print("      (两者输出维度不同，这里只比较「处理相同输入」所需的参数量级)")

# ---------- 5. Feature Map 可视化（真实手写数字） ----------
line("5. Feature Map 可视化（sklearn digits, 8x8）")
digits = load_digits()
img = digits.images[0].astype(np.float32) / 16.0        # (8,8)，归一化到 [0,1]
img_t = torch.from_numpy(img)[None, None, :, :]         # (1,1,8,8)

conv_demo = nn.Conv2d(1, 8, kernel_size=3, padding=1)
with torch.no_grad():
    fmap = torch.relu(conv_demo(img_t))[0]              # (8,8,8)
print(f"输入: {tuple(img_t.shape)}  ->  feature maps: {tuple(fmap.shape)}")
print(f"这一层的参数量: {conv_demo.weight.numel() + conv_demo.bias.numel()}")

fig, axes = plt.subplots(2, 5, figsize=(11, 5.4), constrained_layout=True)
for ax_row in axes:
    for ax in ax_row:
        ax.set_anchor("N")
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title(f"input (8x8)\nlabel = {digits.target[0]}", fontsize=9)
axes[0, 0].axis("off")
for i in range(4):
    axes[0, i + 1].imshow(fmap[i].numpy(), cmap="viridis")
    axes[0, i + 1].set_title(f"filter {i}", fontsize=9)
    axes[0, i + 1].axis("off")
axes[1, 0].imshow(torch.relu(fmap).sum(0).numpy(), cmap="magma")
axes[1, 0].set_title("sum of 8 maps", fontsize=9)
axes[1, 0].axis("off")
for i in range(4):
    axes[1, i + 1].imshow(fmap[i + 4].numpy(), cmap="viridis")
    axes[1, i + 1].set_title(f"filter {i + 4}", fontsize=9)
    axes[1, i + 1].axis("off")
fig.suptitle("Conv2d(1->8, k=3, p=1) feature maps (random weights, not trained)", fontsize=10)
fig.savefig(os.path.join(FIG_DIR, "cnn_feature_maps.png"), bbox_inches="tight")
plt.close(fig)
print("说明：这里权重是随机初始化的，所以这些 map 只是「随机的边缘/斑点检测器」；")
print("      训练之后，前面的层会学到真正的边缘、笔画方向等低级特征。")

# ---------- 6. 感受野 ----------
line("6. Receptive Field（感受野）")
layers = [(3, 1), (3, 1), (3, 2)]     # (kernel, stride)
rf, stride_prod = 1, 1
print(f"{'layer':>6}{'k':>4}{'s':>4}{'RF':>6}")
for i, (k, s) in enumerate(layers, 1):
    rf = rf + (k - 1) * stride_prod
    stride_prod *= s
    print(f"{i:>6}{k:>4}{s:>4}{rf:>6}")
print("\n结论：3 层 3x3 卷积后，一个神经元的感受野是 7x7，")
print("      但每层只有一个 k×k 的 kernel（再乘通道数），参数量不随 H×W 增长。")

print(f"\n图保存至 {os.path.join(FIG_DIR, 'cnn_feature_maps.png')}")

