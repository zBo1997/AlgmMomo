import numpy as np  # 矩阵计算模块

# 代码实现线性回归算法
def h(x):
    return w0 + w1 * x


if __name__ == '__main__':
    # α 步长
    rate = 0.001
    # y = x + 1
    x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_train = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

    # 随机产生w0
    w0 = np.random.normal()
    w1 = np.random.normal()
    err = 1

while err > 0.000000001:
    # 合并两个矩阵
    for x, y in zip(x_train, y_train):
        w0 = w0 - rate * (h(x) - y) * 1  # w0 目标函数
        w1 = w1 - rate * (h(x) - y) * x  # w1 目标函数

    # 计算误差
    err = 0.0
    for x, y in zip(x_train, y_train):
        err += (y - h(x)) ** 2  # y 计算后的值 - h(x) 表示假设的Y值 原有的误差的平方可以计算出误差 （平方误差公式）
    m = len(x_train)  # 样本 数据总量
    err = float(err / (2 * m))  # 再去乘以 1分支2m
    print("err:%f" % err)

print(w0, w1)
