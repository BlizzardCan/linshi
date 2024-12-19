import numpy as np


# 显式Euler法
def explicit_euler(f, y0, t0, t_end, h):
    """
    使用显式Euler法求解常微分方程
    :param f: 常微分方程y' = f(t, y)的函数
    :param y0: 初始值y(t0)
    :param t0: 初始时间
    :param t_end: 结束时间
    :param h: 步长
    :return: 包含在[t0, t_end]区间内，以步长h离散化的y值数组
    """
    t = np.arange(t0, t_end + h, h)  # 创建时间点数组
    y = np.zeros(len(t))  # 初始化y值数组
    y[0] = y0
    for i in range(len(t) - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])  # 显式Euler法的迭代公式
    return y


# 显式梯形法
def explicit_trapezoidal(f, y0, t0, t_end, h):
    """
    使用显式梯形法求解常微分方程
    :param f: 常微分方程y' = f(t, y)的函数
    :param y0: 初始值y(t0)
    :param t0: 初始时间
    :param t_end: 结束时间
    :param h: 步长
    :return: 包含在[t0, t_end]区间内，以步长h离散化的y值数组
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i + 1], y[i] + h * k1)
        y[i + 1] = y[i] + 0.5 * h * (k1 + k2)  # 显式梯形法的迭代公式
    return y


# RK3方法
def rk3(f, y0, t0, t_end, h):
    """
    使用RK3方法求解常微分方程
    :param f: 常微分方程y' = f(t, y)的函数
    :param y0: 初始值y(t0)
    :param t0: 初始时间
    :param t_end: 结束时间
    :param h: 步长
    :return: 包含在[t0, t_end]区间内，以步长h离散化的y值数组
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h/2, y[i] + k1/2)
        k3 = h * f(t[i] + h, y[i] - k1 + 2*k2)
        y[i + 1] = y[i] + (k1 + 4*k2 + k3) / 6  # RK3方法的迭代公式
    return y


# RK4方法
def rk4(f, y0, t0, t_end, h):
    """
    使用RK4方法求解常微分方程
    :param f: 常微分方程y' = f(t, y)的函数
    :param y0: 初始值y(t0)
    :param t0: 初始时间
    :param t_end: 结束时间
    :param h: 步长
    :return: 包含在[t0, t_end]区间内，以步长h离散化的y值数组
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h/2, y[i] + k1/2)
        k3 = h * f(t[i] + h/2, y[i] + k2/2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6  # RK4方法的迭代公式
    return y


# Logistic方程函数
def logistic(t, y):
    """
    Logistic方程：y' = y(1 - y)
    :param t: 时间
    :param y: 函数值
    :return: Logistic方程的导数
    """
    return y * (1 - y)


# 计算误差的函数
def calculate_error(method, h):
    """
    计算给定方法在特定步长下，在t = 2时的误差
    :param method: 求解常微分方程的方法（函数）
    :param h: 步长
    :return: 误差值
    """
    y0 = 0.1
    t0 = 0
    t_end = 2
    y_exact = 1 / (1 + 9 * np.exp(-2))  # Logistic方程在t = 2时的精确解
    y_approx = method(logistic, y0, t0, t_end, h)
    return np.abs(y_approx[-1] - y_exact)


if __name__ == "__main__":
    h_values = [0.1, 0.05, 0.025, 0.0125]
    methods = [explicit_euler, explicit_trapezoidal, rk3, rk4]
    for method in methods:
        print(method.__name__)
        for h in h_values:
            error = calculate_error(method, h)
            print(f"h = {h}, error = {error}")