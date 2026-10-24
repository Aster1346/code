"""
Week 4 Part 4-5: Representation Learning —— Raw vs Learned Representation

实验设计：
1. 训练一个 4-16-3 的 MLP（与 04 完全相同的设置）；
2. 取倒数第二层（ReLU 后的 16 维）作为 Learned Representation；
3. 与 Raw Representation（标准化后的 4 维特征）在三个角度上对比：
   - 余弦相似度：类内 vs 类间（同时给出"未中心化"和"中心化"两个版本）
   - Silhouette Score（cosine 距离）：聚类可分性
   - 1-Nearest-Neighbor 分类准确率：表示是否对下游任务有用
4. t-SNE 二维可视化，直观对比两个空间里的类别结构。

注意：这个脚本特意保留了一个"反直觉"的结果——
      在 Iris 这种小数据上，Learned Representation 不一定比 Raw 更好。
      原因与如何解释，见方案文档 Part 5。

运行：python 06_representation.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 见文档 Part 11 排错表

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- 1. 数据 ----------
iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

X_tr, y_tr = torch.from_numpy(X_train_s), torch.from_numpy(y_train)
X_te, y_te = torch.from_numpy(X_test_s), torch.from_numpy(y_test)

# ---------- 2. 训练 MLP ----------
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
for _ in range(300):
    optimizer.zero_grad()
    loss = criterion(model(X_tr), y_tr)
    loss.backward()
    optimizer.step()
print(f"训练完成，final train loss = {loss.item():.4f}")

# ---------- 3. 提取表示 ----------
def extract_penultimate(model, x_tensor):
    """倒数第二层 = Linear(4,16) + ReLU 的输出。"""
    with torch.no_grad():
        return torch.relu(model[0](x_tensor)).numpy()


feat_train = extract_penultimate(model, X_tr)      # (120, 16) Learned
feat_test = extract_penultimate(model, X_te)       # (30, 16)
raw_train, raw_test = X_train_s, X_test_s          # (120, 4) / (30, 4) Raw

all_raw = np.vstack([raw_train, raw_test])
all_lrn = np.vstack([feat_train, feat_test])
all_y = np.concatenate([y_train, y_test])


# ---------- 4. 度量 ----------
def cosine_stats(feat, labels, center=False):
    """类内 / 类间平均余弦相似度。center=True 时先按维度去均值。"""
    f = feat - feat.mean(axis=0, keepdims=True) if center else feat
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    sim = f @ f.T
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(labels), dtype=bool)
    return sim[same & off_diag].mean(), sim[~same].mean()


def eval_rep(feat_train_, feat_test_, labels_train, labels_test):
    within_raw, between_raw = cosine_stats(np.vstack([feat_train_, feat_test_]), all_y, center=False)
    within_c, between_c = cosine_stats(np.vstack([feat_train_, feat_test_]), all_y, center=True)
    sil = silhouette_score(np.vstack([feat_train_, feat_test_]), all_y, metric="cosine")
    knn = KNeighborsClassifier(n_neighbors=1).fit(feat_train_, labels_train)
    acc = accuracy_score(labels_test, knn.predict(feat_test_))
    return {
        "within_raw": within_raw, "between_raw": between_raw,
        "within_c": within_c, "between_c": between_c,
        "silhouette": sil, "knn_acc": acc,
    }


m_raw = eval_rep(raw_train, raw_test, y_train, y_test)
m_lrn = eval_rep(feat_train, feat_test, y_train, y_test)

print("\n" + "=" * 74)
print("Raw vs Learned Representation（Iris，150 个样本）")
print("=" * 74)
print(f"{'指标':<34}{'Raw (4-d)':>16}{'Learned (16-d)':>18}")
print("-" * 74)
rows = [
    ("维度", 4, 16, "{:>16}{:>18}"),
    ("类内余弦相似度（未中心化）", m_raw["within_raw"], m_lrn["within_raw"], "{:>16.3f}{:>18.3f}"),
    ("类间余弦相似度（未中心化）", m_raw["between_raw"], m_lrn["between_raw"], "{:>16.3f}{:>18.3f}"),
    ("类内余弦相似度（中心化）", m_raw["within_c"], m_lrn["within_c"], "{:>16.3f}{:>18.3f}"),
    ("类间余弦相似度（中心化）", m_raw["between_c"], m_lrn["between_c"], "{:>16.3f}{:>18.3f}"),
    ("Silhouette（cosine，越大越好）", m_raw["silhouette"], m_lrn["silhouette"], "{:>16.3f}{:>18.3f}"),
    ("1-NN Test Accuracy", m_raw["knn_acc"], m_lrn["knn_acc"], "{:>16.3f}{:>18.3f}"),
]
for name, a, b, fmt in rows:
    print(f"{name:<30}" + fmt.format(a, b))

print("\n读法提示：")
print("1. ReLU 的输出全部 >= 0，所以未中心化的余弦相似度会整体偏高（甚至全为正），")
print("   这会掩盖真实结构；对比中心化版本才能看清类内/类间的相对关系。")
print("2. 未中心化的两行中，Learned 的类内/类间相似度同时变大，无法判断好坏；")
print("   看中心化的两行：类内更大（更紧凑）、类间更小（更远离），间隔从约 1.0 变为约 1.2，")
print("   说明类别结构确实更清晰——度量方式的选择会直接改变结论。")
print("3. Silhouette 与 1-NN Accuracy 才是「这个表示对分类有没有用」的直接证据。")

# ---------- 5. t-SNE 可视化 ----------
tsne_raw = TSNE(n_components=2, perplexity=20, random_state=SEED,
                init="pca", learning_rate="auto").fit_transform(all_raw)
tsne_lrn = TSNE(n_components=2, perplexity=20, random_state=SEED,
                init="pca", learning_rate="auto").fit_transform(all_lrn)

palette = ["#1f5fa8", "#b3402a", "#2e7d4f"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, emb, title, sil in [
    (axes[0], tsne_raw, "Raw (standardized 4-D)", m_raw["silhouette"]),
    (axes[1], tsne_lrn, "Learned (16-D penultimate)", m_lrn["silhouette"]),
]:
    for c in range(3):
        mask = all_y == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=26, color=palette[c],
                   label=iris.target_names[c], alpha=0.85,
                   edgecolors="white", linewidths=0.4)
    ax.set_title(f"{title}\nsilhouette(cosine) = {sil:.3f}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="best")
fig.suptitle("t-SNE: raw vs learned representation (150 samples, train + test)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "representation_tsne.png"))
plt.close(fig)

print(f"\n图保存至 {os.path.join(FIG_DIR, 'representation_tsne.png')}")