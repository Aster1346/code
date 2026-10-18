"""
Week 3 Part 4: Polynomial Regression
观察 Model Complexity 与 Generalization 的关系
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("figures", exist_ok=True)

# ---------- 1. 生成带噪声的非线性数据 ----------
rng = np.random.RandomState(42)
n_samples = 80
X = np.sort(rng.rand(n_samples, 1) * 10, axis=0)
y_true = np.sin(X).ravel()
y = y_true + rng.normal(0, 0.3, n_samples)

# ---------- 2. 划分 Training / Validation / Test ----------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

# ---------- 3. 训练不同 degree 的模型 ----------
degrees = [1, 2, 5, 15]
models = {}
results = []

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    models[degree] = model

    results.append({
        "degree": degree,
        "train_mse": mean_squared_error(y_train, model.predict(X_train)),
        "val_mse": mean_squared_error(y_val, model.predict(X_val)),
        "test_mse": mean_squared_error(y_test, model.predict(X_test)),
        "train_mae": mean_absolute_error(y_train, model.predict(X_train)),
        "test_mae": mean_absolute_error(y_test, model.predict(X_test)),
        "train_r2": r2_score(y_train, model.predict(X_train)),
        "test_r2": r2_score(y_test, model.predict(X_test)),
    })

print("=" * 70)
print(f"{'degree':>6} {'train_MSE':>10} {'val_MSE':>10} {'test_MSE':>10} "
      f"{'train_R2':>9} {'test_R2':>9}")
print("=" * 70)
for r in results:
    print(f"{r['degree']:>6} {r['train_mse']:>10.4f} {r['val_mse']:>10.4f} "
          f"{r['test_mse']:>10.4f} {r['train_r2']:>9.3f} {r['test_r2']:>9.3f}")

# ---------- 4. 绘制每个 degree 的拟合曲线 ----------
X_plot = np.linspace(0, 10, 500).reshape(-1, 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.ravel()

for ax, degree in zip(axes, degrees):
    y_plot = models[degree].predict(X_plot)
    ax.scatter(X_train, y_train, s=20, label="Train", alpha=0.7)
    ax.scatter(X_test, y_test, s=20, label="Test", alpha=0.7)
    ax.plot(X_plot, np.sin(X_plot), "k--", label="True sin(x)", linewidth=1)
    ax.plot(X_plot, y_plot, "r-", label=f"degree={degree}", linewidth=2)
    ax.set_title(f"Degree = {degree}")
    ax.set_ylim(-2.5, 2.5)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/polynomial_fit_curves.png", dpi=150)
plt.close()

# ---------- 5. Training vs Test Error ----------
degrees_list = [r["degree"] for r in results]
train_mse = [r["train_mse"] for r in results]
test_mse = [r["test_mse"] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(degrees_list, train_mse, "o-", label="Train MSE")
plt.plot(degrees_list, test_mse, "s-", label="Test MSE")
plt.xlabel("Degree")
plt.ylabel("MSE (log scale)")
plt.yscale("log")
plt.title("Training vs Test Error for Different Degrees")
plt.xticks(degrees_list)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figures/train_test_error.png", dpi=150)
plt.close()

# ---------- 6. degree 1~20 扫描 ----------
train_errors, test_errors = [], []
deg_range = range(1, 21)

for d in deg_range:
    model = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(list(deg_range), train_errors, "o-", label="Train MSE")
plt.plot(list(deg_range), test_errors, "s-", label="Test MSE")
plt.xlabel("Degree")
plt.ylabel("MSE (log scale)")
plt.yscale("log")
plt.title("Model Complexity vs Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figures/degree_vs_error.png", dpi=150)
plt.close()

print("\nFigures saved to figures/")