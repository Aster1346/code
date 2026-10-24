"""
Week 4 Part 4: 改变 Hidden Dimension（8 / 16 / 64）对比实验

科研习惯要点：
1. 只改一个变量（hidden dimension），其他全部固定；
2. 每个配置跑多个随机种子，报告 mean ± std，而不是只挑一个好看的结果；
3. 同时记录参数量与训练时间，解释"模型更大"的代价。

运行：python 05_hidden_dim_study.py
"""

import csv
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
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

# ---------- 1. 数据（固定划分，所有配置共用） ----------
iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_t = torch.from_numpy(scaler.transform(X_train).astype(np.float32))
X_test_t = torch.from_numpy(scaler.transform(X_test).astype(np.float32))
y_train_t = torch.from_numpy(y_train)
y_test_t = torch.from_numpy(y_test)


def build_model(hidden_dim):
    return nn.Sequential(
        nn.Linear(4, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 3),
    )


def run_once(hidden_dim, seed, epochs=300, lr=0.01):
    torch.manual_seed(seed)
    model = build_model(hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    t0 = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).argmax(1).numpy()
        test_pred = model(X_test_t).argmax(1).numpy()

    return {
        "hidden_dim": hidden_dim,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "final_train_loss": losses[-1],
        "train_acc": accuracy_score(y_train, train_pred),
        "test_acc": accuracy_score(y_test, test_pred),
        "train_time_s": train_time,
        "losses": losses,
    }


# ---------- 2. 跑实验 ----------
HIDDEN_DIMS = [8, 16, 64]
SEEDS = [0, 1, 2, 3, 4]

rows = []
for h in HIDDEN_DIMS:
    for seed in SEEDS:
        rows.append(run_once(h, seed))
    accs = [r["test_acc"] for r in rows if r["hidden_dim"] == h]
    print(f"hidden={h:>2}  完成 {len(accs)} 个种子")

# ---------- 3. 汇总表 ----------
print("\n" + "=" * 78)
print(f"{'hidden':>7} {'params':>8} {'train_loss':>11} {'train_acc':>10} "
      f"{'test_acc (mean±std)':>20} {'time(s)':>8}")
print("=" * 78)
summary = []
for h in HIDDEN_DIMS:
    sub = [r for r in rows if r["hidden_dim"] == h]
    accs = np.array([r["test_acc"] for r in sub])
    losses = np.array([r["final_train_loss"] for r in sub])
    item = {
        "hidden_dim": h,
        "params": sub[0]["params"],
        "train_loss_mean": losses.mean(),
        "train_loss_std": losses.std(ddof=1),
        "train_acc_mean": np.mean([r["train_acc"] for r in sub]),
        "test_acc_mean": accs.mean(),
        "test_acc_std": accs.std(ddof=1),
        "time_mean": np.mean([r["train_time_s"] for r in sub]),
    }
    summary.append(item)
    print(f"{h:>7} {item['params']:>8} {item['train_loss_mean']:>11.4f} "
          f"{item['train_acc_mean']:>10.4f} "
          f"{item['test_acc_mean']:>9.4f} ± {item['test_acc_std']:.4f} "
          f"{item['time_mean']:>8.2f}")

# ---------- 4. 保存逐次结果 ----------
csv_path = os.path.join(RES_DIR, "hidden_dim_results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["hidden_dim", "seed", "params", "final_train_loss",
                     "train_acc", "test_acc", "train_time_s"])
    for r in rows:
        writer.writerow([r["hidden_dim"], r["seed"], r["params"],
                         round(r["final_train_loss"], 6), round(r["train_acc"], 4),
                         round(r["test_acc"], 4), round(r["train_time_s"], 3)])

# ---------- 5. 图 ----------
colors = {8: "#8fb7d9", 16: "#1f5fa8", 64: "#0d2f5c"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
for h in HIDDEN_DIMS:
    curves = np.array([r["losses"] for r in rows if r["hidden_dim"] == h])
    mean, std = curves.mean(0), curves.std(0, ddof=1)
    epochs = np.arange(len(mean))
    ax.plot(epochs, mean, color=colors[h], lw=1.8, label=f"hidden = {h}")
    ax.fill_between(epochs, mean - std, mean + std, color=colors[h], alpha=0.18, lw=0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_yscale("log")
ax.set_title("Loss curves (mean ± std over 5 seeds)")
ax.legend()

ax = axes[1]
means = [s["test_acc_mean"] for s in summary]
stds = [s["test_acc_std"] for s in summary]
ax.bar([str(h) for h in HIDDEN_DIMS], means, yerr=stds, capsize=5,
       color=[colors[h] for h in HIDDEN_DIMS], width=0.55, alpha=0.9)
for i, h in enumerate(HIDDEN_DIMS):
    pts = [r["test_acc"] for r in rows if r["hidden_dim"] == h]
    ax.scatter([i] * len(pts), pts, s=14, c="k", alpha=0.55, zorder=3)
ax.set_ylim(0.6, 1.03)
ax.set_xlabel("Hidden dimension")
ax.set_ylabel("Test Accuracy")
ax.set_title("Test accuracy: bars = mean ± std, dots = each seed")

fig.suptitle("Iris MLP: effect of hidden dimension (other settings fixed)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "hidden_dim_comparison.png"))
plt.close(fig)

print(f"\nCSV 保存至 {csv_path}")
print(f"图保存至 {os.path.join(FIG_DIR, 'hidden_dim_comparison.png')}")
print("\n读图提示：Iris 只有 120 个训练样本、4 个特征，")
print("          隐藏层从 16 加到 64（参数量几乎翻倍）并不会稳定提升 Test Accuracy，")
print("          说明「模型更大」不等于「泛化更好」——这正是 Week 3 Overfitting 的神经网络版本。")