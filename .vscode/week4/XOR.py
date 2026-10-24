import os

# Windows/conda 里 torch 和 numpy(MKL) 各带一份 OpenMP，可能冲突报 "OMP: Error #15"。
# 这行必须在 import torch 之前执行（原因见方案文档 Part 11 的排错表）。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

torch.manual_seed(0)
np.random.seed(0)

# ---------- 1. 构造 XOR 数据（每个点重复 100 次并加轻微噪声，便于训练稳定） ----------
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

n_per_class = 100
idx = torch.randint(0, 4, (n_per_class * 4,))
X_data = X[idx] + 0.12 * torch.randn(n_per_class * 4, 2)   # 加噪声避免完全可分
y_data = y[idx]


# ---------- 2. 两个模型 ----------
linear_model = nn.Linear(2, 2)                    # 等价于多类 Logistic Regression
mlp_model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)


def train(model, epochs=600, lr=0.05):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_data)
        loss = criterion(logits, y_data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


loss_linear = train(linear_model)
loss_mlp = train(mlp_model)


# ---------- 3. 画决策边界 ----------
def decision_grid(model, n=200):
    xs = np.linspace(-0.6, 1.6, n)
    ys = np.linspace(-0.6, 1.6, n)
    gx, gy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        pred = model(grid).argmax(dim=1).numpy().reshape(gx.shape)
    return gx, gy, pred


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, model, title in [
    (axes[0], linear_model, "Logistic Regression (linear): accuracy = %.2f" % (
        (linear_model(X_data).argmax(1) == y_data).float().mean().item())),
    (axes[1], mlp_model, "MLP (1 hidden layer, ReLU): accuracy = %.2f" % (
        (mlp_model(X_data).argmax(1) == y_data).float().mean().item())),
]:
    gx, gy, pred = decision_grid(model)
    ax.contourf(gx, gy, pred, levels=[-0.5, 0.5, 1.5], colors=["#cfe3f7", "#f7d9cf"], alpha=0.85)
    ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1], s=10, c="#1f5fa8", label="class 0")
    ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1], s=10, c="#b3402a", label="class 1")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("XOR: a linear model can only draw one straight boundary", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "xor_linear_vs_mlp.png"))
plt.close(fig)

# ---------- 4. 打印对照结果 ----------
acc_linear = (linear_model(X_data).argmax(1) == y_data).float().mean().item()
acc_mlp = (mlp_model(X_data).argmax(1) == y_data).float().mean().item()

print("=" * 56)
print("XOR 实验（训练集准确率）")
print("=" * 56)
print(f"Logistic Regression : acc = {acc_linear:.4f}  final loss = {loss_linear[-1]:.4f}")
print(f"MLP (2-16-2)        : acc = {acc_mlp:.4f}  final loss = {loss_mlp[-1]:.4f}")
print(f"\nFigure saved to {os.path.join(FIG_DIR, 'xor_linear_vs_mlp.png')}")
print("\n观察：线性模型的边界是一条直线。直线最多只能把 4 个角点里的 3 个分到正确一侧，")
print("      所以准确率上限约为 0.75（本例中的类 0 = {(0,0), (1,1)} 无法被一条直线包住）；")
print("      MLP 用 ReLU 折出多段边界，把两个对角区域包了起来，才能到 1.00。")
