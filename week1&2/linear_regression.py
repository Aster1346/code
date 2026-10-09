"""线性回归的批量梯度下降训练循环"""

import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据用以训练
np.random.seed(42)
x = np.linspace(-5, 5, 100)
noise = np.random.normal(0, 1, size=100)
y = 2 * x + 1 + noise

# 初始化参数
w = 0.0
b = 0.0
lr = 0.01
epochs = 1000#训练轮数
loss_history = []#误差记录

"""
通过损失函数分别对w和b求偏导，得到梯度计算公式如下
∂L/∂w = (2/N) Σ xᵢ(ŷᵢ - yᵢ)
∂L/∂b = (2/N) Σ(ŷᵢ - yᵢ)
"""

for epoch in range(epochs):
    y_pred = w * x + b
    loss = np.mean((y_pred - y) ** 2)#Mean Squared Error均方误差

    dw = 2 * np.mean(x * (y_pred - y))
    db = 2 * np.mean(y_pred - y)

    #梯度下降更新参数
    w = w - lr * dw
    b = b - lr * db

    loss_history.append(loss)

print("w =", w)
print("b =", b)
print("final loss =", loss_history[-1])
print("真实参数: w_true = 2, b_true = 1")

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.6, label="noisy data")
plt.plot(x, w * x + b, color="red", linewidth=2,
         label=f"pred: y = {w:.3f}x + {b:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Training Loss")
plt.tight_layout()
plt.show()
