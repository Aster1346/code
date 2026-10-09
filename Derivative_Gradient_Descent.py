"""考虑函数：
y = (x - 3)²
其导数为：
dy/dx = 2(x - 3)
Gradient Descent 的基本更新形式：
x(t+1) = x(t) - η · dy/dx"""


def gradient_descent(x_start, learning_rate, num_iterations):
    x = x_start
    for step in range(num_iterations):
        gradient = 2 * (x - 3)
        x = x - learning_rate * gradient
        print(step, x)
    return x
x = 0.0
lr = 0.1
gradient_descent(x, lr, 20)
#changing learning rate
lr = 0.01
gradient_descent(x, lr, 20)
#changing learning rate
lr = 0.001
gradient_descent(x, lr, 20)
#changing learning rate
lr = 1
gradient_descent(x, lr, 20)
#changing learning rate
lr = 1.1
gradient_descent(x, lr, 20)

"""
learning rate太大，梯度下降会严重发散，调1.1后设置有112到-93的误差，无法收敛到最优解
learning rate太小，梯度下降会收敛的很慢，调0.01后设置有20步才收敛到最优解
"""