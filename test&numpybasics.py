import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-5,5,100)
y = x ** 2
plt.plot(x,y)
plt.show()

#一维数组:向量
a = np.array([1,2,3])
print(a)

#二维数组：矩阵
A = np.array([[1,2,3],
             [4,5,6]])
print(A)

#全零数组
Z = np.zeros((2,3))#两行三列
print(Z)

#全一数组
O = np.ones((3,4))#三行四列
print(O)

#指定值矩阵
f = np.full((2,3),7)#两行三列,7
print(f)

r = np.arange(0,10,2)#[0,2,4,6,8]
l = np.linspace(0,1,5)#[0,0.25,0.5,0.75,1]

#随机数
rand = np.random.rand(2,3)#两行三列，0到1随机小数
randint = np.random.randint(0,10,(2,3))#两行三列，0到9随机整数
noise = np.random.normal(0, 1, size=100)#100个服从正态分布的随机数，模拟误差

"""查看数组属性"""
A = np.array([[1,2,3],
             [4,5,6]])
print(A.shape)#(2,3) 两行三列
print(A.ndim)#2 维度
print(A.size)#6 元素个数
print(A.dtype)#int64数据类型

"""向量计算"""
u = np.array([1,2,3])
v = np.array([4,5,6])
print(u+v)#向量加法[5,7,9]
print(u-v)#向量减法[-3,-3,-3]
print(u*v)#向量逐元素相乘[4,10,18]
print(u/v)#向量逐元素相除[0.25,0.4,0.5]
print(u*10)#向量数乘[10,20,30]
print(np.dot(u,v))#向量点积32
print(u@v)#向量点积32
print(np.linalg.norm(u))#向量模3.7416573867739413

#统计
print(np.mean(u))#向量均值2.0
   