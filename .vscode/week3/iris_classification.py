"""
Week 3 Part 6-2: Iris Classification
使用 Logistic Regression 完成三分类，输出 Accuracy、F1、Confusion Matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

# ---------- 路径 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 1. 加载数据 ----------
iris = load_iris()
X, y = iris.data, iris.target

# ---------- 2. 划分 Training / Test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ---------- 3. Pipeline：标准化 + Logistic Regression ----------
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# ---------- 4. 训练 ----------
model.fit(X_train, y_train)

# ---------- 5. 预测 ----------
y_pred = model.predict(X_test)

# ---------- 6. 指标 ----------
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("\n=== 指标 ===")
print(f"Accuracy : {acc:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")
print("\nConfusion Matrix:")
print(cm)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ---------- 7. 保存混淆矩阵图 ----------
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=iris.target_names,
    cmap="Blues",
    ax=ax
)
ax.set_title("Iris Logistic Regression - Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "iris_confusion_matrix.png"), dpi=150)
plt.close(fig)

print(f"\nFigure saved to {os.path.join(FIG_DIR, 'iris_confusion_matrix.png')}")