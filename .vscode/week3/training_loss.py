import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. 生成带噪声的非线性数据
rng = np.random.RandomState(42)
X = np.sort(rng.rand(60, 1) * 10, axis=0)
y = np.sin(X).ravel() + rng.normal(0, 0.3, X.shape[0])

# 2. 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
#degree,多项式拟合阶数
# 3. 比较不同 degree
for degree in [1, 2, 5, 15]:
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"degree={degree:2d}  train_MSE={train_mse:.4f}  test_MSE={test_mse:.4f}")
    #样例输出
    #degree= 1  train_MSE=0.4586  test_MSE=0.4649
    #degree= 2  train_MSE=0.3957  test_MSE=0.4644
    #degree= 5  train_MSE=0.0950  test_MSE=0.0826  最佳
    #degree=15  train_MSE=0.4644  test_MSE=0.3211