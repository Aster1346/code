"""
Week 4 Part 2: PyTorch Autograd 基础实验

覆盖：
1. requires_grad / backward() / .grad 的最小例子，并与手算导数核对；
2. 梯度是累加的（backward 调用两次会变成 2 倍），所以每个 step 前要 zero_grad；
3. grad_fn 计算图，detach 与 torch.no_grad() 的区别；
4. 用 autograd 重做 Week 2 的线性回归，和解析解对比；
5. CrossEntropyLoss = KL Divergence（当标签是 one-hot 时），接上 Week 3 的 KL。
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 见文档 Part 11 排错表

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


# ---------- 1. 最小例子：y = sum(x^2)，dy/dx = 2x ----------
line("1. 最小 Autograd 例子")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
print(f"x            = {x}")
print(f"y = sum(x^2) = {y}")
print(f"y.grad_fn    = {y.grad_fn}")

y.backward()
print(f"x.grad       = {x.grad}          <- 2x = [2, 4, 6]")
print(f"手算 2x      = {2 * x.detach()}")
print(f"是否一致      : {torch.allclose(x.grad, 2 * x.detach())}")
print(f"x.grad_fn    = {x.grad_fn}       <- 叶子节点没有 grad_fn")
print(f"x.is_leaf    = {x.is_leaf}")


# ---------- 2. 梯度累加 ----------
line("2. 梯度累加：backward() 不会自动清零")
x2 = torch.tensor([1.0, 2.0], requires_grad=True)
y2 = (x2 ** 2).sum()
y2.backward(retain_graph=True)          # retain_graph=True 保留计算图，才能再 backward 一次
print(f"第一次 backward 后 x2.grad = {x2.grad}")
y2.backward()
print(f"第二次 backward 后 x2.grad = {x2.grad}   <- 累加变成 2 倍")
x2.grad.zero_()
print(f"zero_() 之后         x2.grad = {x2.grad}")

# 如果不加 retain_graph，第二次 backward 会直接报错（图已被释放）
x3 = torch.tensor([1.0], requires_grad=True)
y3 = (x3 ** 2).sum()
y3.backward()
try:
    y3.backward()
except RuntimeError as err:
    print(f"\n不加 retain_graph 时第二次 backward 报错:\n  {type(err).__name__}: {str(err)[:90]}...")
print("=> 训练循环里每轮都重新 forward 一次，所以不需要 retain_graph；")
print("   但 'zero_grad() 在 backward 之前' 这个顺序仍然不能省。")


# ---------- 3. 计算图、detach、no_grad ----------
line("3. 计算图与 detach / no_grad")
a = torch.tensor([2.0], requires_grad=True)
b = a * 3
c = b ** 2
print(f"b.grad_fn = {b.grad_fn}")
print(f"c.grad_fn = {c.grad_fn}")
print(f"c.requires_grad = {c.requires_grad}")

d = c.detach()                          # 共享内存，但脱离计算图
print(f"d.requires_grad = {d.requires_grad}   <- detach 后不再求导")
print(f"d.data_ptr == c.data_ptr : {d.data_ptr() == c.data_ptr()}")

with torch.no_grad():                   # 上下文内所有运算都不建图
    e = a * 3
print(f"no_grad 内 e.requires_grad = {e.requires_grad}")

# 非标量输出必须指定 gradient 参数
line("3b. 非标量输出必须指定 gradient")
v = torch.tensor([1.0, 2.0], requires_grad=True)
out = v * 2
try:
    out.backward()
except RuntimeError as err:
    print("直接 backward() 报错:", err)
out.backward(torch.tensor([1.0, 1.0]))   # 等价于对 sum(out) 求导
print(f"指定 gradient 后 v.grad = {v.grad}   <- 2 * [1, 1] = [2, 2]")


# ---------- 4. 用 autograd 重做线性回归 ----------
line("4. Autograd 版线性回归 vs 解析解")
torch.manual_seed(1)
N = 50
x_data = torch.rand(N, 1) * 4
true_w, true_b = 2.5, -1.0
y_data = true_w * x_data + true_b + 0.3 * torch.randn(N, 1)

w = torch.zeros(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.05
for step in range(301):
    y_pred = x_data @ w + b
    loss = ((y_pred - y_data) ** 2).mean()
    loss.backward()
    with torch.no_grad():                # 更新参数时不要建图
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

# 解析解（最小二乘，Week 2 的公式）
X_aug = torch.cat([x_data, torch.ones(N, 1)], dim=1)
theta = torch.linalg.lstsq(X_aug, y_data).solution
print(f"梯度下降结果 : w = {w.item():.4f}, b = {b.item():.4f}, final loss = {loss.item():.4f}")
print(f"解析解       : w = {theta[0].item():.4f}, b = {theta[1].item():.4f}")
print("说明：两种方法应当收敛到接近的参数；autograd 只负责求梯度，")
print("      参数怎么更新仍由 'w -= lr * w.grad' 这一步决定。")


# ---------- 5. CrossEntropyLoss 与 KL Divergence ----------
line("5. CrossEntropyLoss == KL Divergence（one-hot 标签）")
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.2, 1.5, 2.2]])
labels = torch.tensor([0, 1])

ce = F.cross_entropy(logits, labels)
probs = F.softmax(logits, dim=1)
print(f"logits =\n{logits}")
print(f"softmax(logits) =\n{probs}")
print(f"每行概率和 = {probs.sum(dim=1)}")

# 手写 Cross Entropy：-log q_y
q_y = probs[torch.arange(len(labels)), labels]
manual_ce = (-torch.log(q_y)).mean()
print(f"nn.CrossEntropyLoss = {ce:.6f}")
print(f"手写 -log(q_y)      = {manual_ce:.6f}")

# 用 KL 公式：D_KL(p||q) = sum p * log(p/q)，p 为 one-hot
p = F.one_hot(labels, num_classes=3).float()
kl_per_sample = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(probs))).sum(dim=1)
print(f"逐样本 KL(p||q)     = {kl_per_sample.detach()}")
print(f"KL 的平均值 + H(p)  = {kl_per_sample.mean().item():.6f}  (one-hot 时 H(p)=0)")
print("\n结论：H(p, q) = H(p) + D_KL(p||q)；当标签是 one-hot 时 H(p)=0，")
print("      所以 CrossEntropyLoss 就是预测分布与真实分布之间的 KL Divergence。")
print("      Week 3 的 KL 不是另一个知识点，它正是本周分类 Loss 的本质。")