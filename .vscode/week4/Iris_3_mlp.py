"""
Week 4 Part 3-4: Iris 三分类 —— PyTorch MLP 完整训练流程

对应本周实验要求 1：
- 复用 Week 3 的数据纪律：stratified split、StandardScaler 只在 Training Set 上 fit；
- 训练 4-16-3 的 MLP，记录 Training Loss 与 Test Accuracy；
- 绘制 Loss Curve，并输出 Accuracy / F1 / Confusion Matrix；
- 与 Week 3 的 Logistic Regression 基线对照。

运行：python 04_iris_mlp.py
"""

import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 见文档 Part 11 排错表

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- 1. 数据：与 Week 3 完全相同的划分 ----------
iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler().fit(X_train)            # 只在训练集上 fit
X_train_s = scaler.transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

print(f"Training: {X_train_s.shape}   Test: {X_test_s.shape}")

# 转成 Tensor：X 用 float32，标签必须是 int64（CrossEntropyLoss 要求类别索引）
X_train_t = torch.from_numpy(X_train_s)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test_s)
y_test_t = torch.from_numpy(y_test)

# ---------- 2. 模型 ----------
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)
print(f"\n模型结构:\n{model}")
print(f"参数量: {sum(p.numel() for p in model.parameters())}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- 3. Training Loop ----------
EPOCHS = 300
history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()

    optimizer.zero_grad()                  # 1) 清空上一步的梯度
    logits = model(X_train_t)              # 2) Forward：得到 logits（未过 softmax）
    loss = criterion(logits, y_train_t)    # 3) 算 Loss
    loss.backward()                        # 4) Backward：autograd 求每个参数的梯度
    optimizer.step()                       # 5) 用梯度更新参数

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t).argmax(dim=1)
            test_pred = model(X_test_t).argmax(dim=1)
            train_acc = (train_pred == y_train_t).float().mean().item()
            test_acc = (test_pred == y_test_t).float().mean().item()
        history["epoch"].append(epoch)
        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:>3}  loss {loss.item():.4f}  "
                  f"train_acc {train_acc:.3f}  test_acc {test_acc:.3f}")

train_time = time.time() - t0

# ---------- 4. 评估 ----------
model.eval()
with torch.no_grad():
    logits_test = model(X_test_t)
    probs_test = torch.softmax(logits_test, dim=1)
    y_pred = logits_test.argmax(dim=1).numpy()

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 56)
print("Test 结果（30 个样本）")
print("=" * 56)
print(f"Accuracy  : {acc:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"\n前 3 个样本的预测概率:\n{probs_test[:3].numpy().round(3)}")

# ---------- 5. Week 3 基线对照 ----------
baseline = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
base_acc = accuracy_score(y_test, baseline.predict(X_test_s))
base_f1 = f1_score(y_test, baseline.predict(X_test_s), average="macro")
print(f"\nWeek 3 Logistic Regression 基线: acc = {base_acc:.4f}, f1 = {base_f1:.4f}")

# ---------- 6. 保存图 ----------
fig, ax1 = plt.subplots(figsize=(8, 4.8))
ax1.plot(history["epoch"], history["train_loss"], color="#1f5fa8", lw=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss (CrossEntropy)", color="#1f5fa8")
ax1.tick_params(axis="y", labelcolor="#1f5fa8")

ax2 = ax1.twinx()
ax2.plot(history["epoch"], history["test_acc"], color="#b3402a", lw=1.6, ls="--")
ax2.set_ylabel("Test Accuracy", color="#b3402a")
ax2.tick_params(axis="y", labelcolor="#b3402a")
ax2.set_ylim(0, 1.05)
ax2.grid(False)

ax1.set_title("Iris MLP (4-16-3): Training Loss and Test Accuracy")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "loss_curve_iris.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 4.6))
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(
    ax=ax, cmap="Blues", colorbar=False, values_format="d"
)
ax.set_title("Iris MLP - Confusion Matrix (test set)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "iris_mlp_confusion_matrix.png"))
plt.close(fig)

# ---------- 7. 保存结果 ----------
metrics = {
    "model": "MLP 4-16-3, ReLU, Adam lr=0.01, full-batch, 300 epochs, seed=42",
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "params": int(sum(p.numel() for p in model.parameters())),
    "train_time_s": round(train_time, 2),
    "final_train_loss": float(history["train_loss"][-1]),
    "test_accuracy": float(acc),
    "test_f1_macro": float(f1_macro),
    "confusion_matrix": cm.tolist(),
    "baseline_logreg_accuracy": float(base_acc),
    "baseline_logreg_f1": float(base_f1),
}
with open(os.path.join(RES_DIR, "iris_mlp_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\n图表保存至 {FIG_DIR}")
print(f"指标保存至 {os.path.join(RES_DIR, 'iris_mlp_metrics.json')}")