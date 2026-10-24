import numpy as np

np.set_printoptions(precision=6, suppress=True)


# ---------- 0. 激活函数 ----------
def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    # ReLU'(z) = 1 if z > 0 else 0
    return (z > 0).astype(float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)


# ---------- 1. 网络参数（固定，便于手算核对） ----------
x = np.array([1.0, 2.0])          # 输入 (2,)
y = 1.0                            # 目标标签（二分类，1 表示正类）

W1 = np.array([[0.5, -0.3],
               [0.2,  0.4]])       # (2, 2)：输出 2 个 hidden unit
b1 = np.array([0.2, -0.05])        # (2,)

W2 = np.array([[0.7, -0.4]])       # (1, 2)：输出 1 个 logit
b2 = np.array([0.05])              # (1,)
# 注意：这里特意让 z1 的每个元素都不等于 0。
# 如果某个 z1 恰好等于 0，就落在 ReLU 的折点上，ReLU'(0) 没有唯一取值，
# 数值梯度会给出 0 和 1 之间的"平均斜率"，梯度检查就会误报失败。


# ---------- 2. Forward Propagation ----------
z1 = W1 @ x + b1                   # (2,)
a1 = relu(z1)                      # (2,) hidden activation
z2 = W2 @ a1 + b2                  # (1,)
y_hat = sigmoid(z2)[0]             # 标量，预测概率

loss = 0.5 * (y_hat - y) ** 2      # 用平方误差，方便手算；真实任务常用 BCE / CrossEntropy

print("=" * 64)
print("Forward Propagation")
print("=" * 64)
print(f"x      = {x}")
print(f"z1     = W1 @ x + b1 = {z1}")
print(f"a1     = relu(z1)    = {a1}")
print(f"z2     = W2 @ a1 + b2 = {z2}")
print(f"y_hat  = sigmoid(z2)  = {y_hat:.6f}")
print(f"loss   = 0.5*(y_hat-y)^2 = {loss:.6f}")


# ---------- 3. Backward Propagation（链式法则逐层回传） ----------
# dL/dy_hat
d_y_hat = (y_hat - y)                                  # 标量

# y_hat = sigmoid(z2)  =>  dL/dz2 = dL/dy_hat * sigmoid'(z2)
d_z2 = d_y_hat * sigmoid_grad(z2)                      # (1,)

# z2 = W2 @ a1 + b2
d_W2 = np.outer(d_z2, a1)                              # (1,2)
d_b2 = d_z2.copy()                                     # (1,)
d_a1 = W2.T @ d_z2                                     # (2,)

# a1 = relu(z1)
d_z1 = d_a1 * relu_grad(z1)                            # (2,)

# z1 = W1 @ x + b1
d_W1 = np.outer(d_z1, x)                               # (2,2)
d_b1 = d_z1.copy()                                     # (2,)

print()
print("=" * 64)
print("Backward Propagation（解析梯度）")
print("=" * 64)
print(f"dL/dz2 = {d_z2}")
print(f"dL/dW2 = \n{d_W2}")
print(f"dL/db2 = {d_b2}")
print(f"dL/da1 = {d_a1}")
print(f"dL/dz1 = {d_z1}")
print(f"dL/dW1 = \n{d_W1}")
print(f"dL/db1 = {d_b1}")


# ---------- 4. 数值梯度检查 ----------
def loss_from_params(W1_, b1_, W2_, b2_, eps=0.0, idx=None):
    """给定参数算 loss；eps/idx 用于数值梯度扰动。"""
    W1c, b1c, W2c, b2c = W1_.copy(), b1_.copy(), W2_.copy(), b2_.copy()
    if idx is not None:
        arr, i = idx
        if arr == "W1":
            W1c[i] += eps
        elif arr == "b1":
            b1c[i] += eps
        elif arr == "W2":
            W2c[i] += eps
        else:
            b2c[i] += eps
    h = relu(W1c @ x + b1c)
    out = sigmoid(W2c @ h + b2c)[0]
    return 0.5 * (out - y) ** 2


def numerical_grad():
    eps = 1e-6
    grads = {}
    for name, param, analytic in [
        ("W1", W1, d_W1), ("b1", b1, d_b1),
        ("W2", W2, d_W2), ("b2", b2, d_b2),
    ]:
        g = np.zeros_like(param)
        for i in np.ndindex(param.shape):
            plus = loss_from_params(W1, b1, W2, b2, +eps, (name, i))
            minus = loss_from_params(W1, b1, W2, b2, -eps, (name, i))
            g[i] = (plus - minus) / (2 * eps)
        grads[name] = (g, analytic)
    return grads


print()
print("=" * 64)
print("梯度检查：解析梯度 vs 数值梯度")
print("=" * 64)
print(f"{'param':>5} {'shape':>8} {'max_abs_diff':>14} {'max_rel_diff':>14}")
print("-" * 64)
worst = 0.0
for name, (num, ana) in numerical_grad().items():
    abs_diff = np.max(np.abs(num - ana))
    rel_diff = abs_diff / max(1e-8, np.max(np.abs(num)) + np.max(np.abs(ana)))
    worst = max(worst, rel_diff)
    print(f"{name:>5} {str(ana.shape):>8} {abs_diff:>14.3e} {rel_diff:>14.3e}")
print("-" * 64)
print(f"最大相对误差 = {worst:.3e}  ->  {'PASS' if worst < 1e-5 else 'FAIL'}（经验阈值 1e-5）")


# ---------- 5. 反例：没有非线性，多层线性 = 一层线性 ----------
# 注意：这里特意不带激活函数，模拟 "多个 Linear Layer 直接串联" 的情况。
W1_lin = np.array([[0.5, -0.3],
                   [0.2,  0.8]])
b1_lin = np.array([0.1, -0.1])
W2_lin = np.array([[0.7, -0.4],
                   [0.1,  0.9]])
b2_lin = np.array([0.05, -0.2])

W_eff = W2_lin @ W1_lin                      # (2,2)
b_eff = W2_lin @ b1_lin + b2_lin              # (2,)

X_test = np.array([[1.0, 2.0],
                   [-1.0, 0.5],
                   [3.0, -2.0],
                   [0.0, 0.0]])

two_layers = (X_test @ W1_lin.T + b1_lin) @ W2_lin.T + b2_lin
one_layer = X_test @ W_eff.T + b_eff

print()
print("=" * 64)
print("两个 Linear Layer（无激活）vs 一个 Linear Layer")
print("=" * 64)
print(f"W_eff = W2 @ W1 = \n{W_eff}")
print(f"b_eff = W2 @ b1 + b2 = {b_eff}")
print(f"两层输出 = \n{two_layers}")
print(f"一层输出 = \n{one_layer}")
print(f"最大差异 = {np.max(np.abs(two_layers - one_layer)):.3e}")
print("\n结论：W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2)，仍然只是一个仿射变换。")
print("      无论堆多少层线性层，表达能力都等价于单层线性模型；")
print("      加 ReLU 之后才能拟合 XOR 这类非线性决策边界。")

