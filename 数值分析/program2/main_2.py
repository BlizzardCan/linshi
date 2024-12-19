import numpy as np

# 定义被积函数 f(x) = xe^x
def f(x):
    return x * np.exp(x)

# 复合中心法则
def composite_midpoint(f, a, b, n):
    h = (b - a) / n
    x = a + h / 2 + np.arange(n) * h
    return h * np.sum(f(x))

# 复合梯形法则
def composite_trapezoid(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))

# 复合 Simpson 法则
def composite_simpson(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h / 3 * (f(x[0]) + 4 * np.sum(f(x[1::2])) + 2 * np.sum(f(x[2::2])) + f(x[-1]))

# 复合 3 点 Gauss 积分
def composite_gauss3(f, a, b, n):
    h = (b - a) / n
    x = a + h * (0.5 + np.arange(n) - 0.5 * np.array([1, 0, -1]))
    w = h * np.array([5/9, 8/9, 5/9])
    return np.sum(w * f(x))

# 复合 3 点 Gauss 积分
def composite_gauss3(f, a, b, n):
    # Gauss 积分的节点和权重
    x1, x2, x3 = -np.sqrt(3)/3, 0, np.sqrt(3)/3
    w1, w2, w3 = 5/9, 8/9, 5/9
    # 计算每个子区间的宽度
    h = (b - a) / n
    # 初始化积分近似值
    integral_approx = 0
    
    # 对每个子区间进行积分
    for i in range(n):
        # 计算子区间的节点位置
        nodes = a + (i * h) + h * np.array([x1, x2, x3])
        # 计算权重与函数值的乘积之和
        integral_approx += h * (w1 * f(nodes[0]) + w2 * f(nodes[1]) + w3 * f(nodes[2]))
    
    return integral_approx

# 正确积分值
exact_integral = 1

# 子区间数
n_values = [1, 2, 4, 8, 16, 32]
errors = {method: [] for method in ['Midpoint', 'Trapezoid', 'Simpson', 'Gauss3']}

for n in n_values:
    midpoint_approx = composite_midpoint(f, 0, 1, n)
    trapezoid_approx = composite_trapezoid(f, 0, 1, n)
    simpson_approx = composite_simpson(f, 0, 1, n)
    gauss3_approx = composite_gauss3(f, 0, 1, n)
    
    errors['Midpoint'].append(np.abs(midpoint_approx - exact_integral))
    errors['Trapezoid'].append(np.abs(trapezoid_approx - exact_integral))
    errors['Simpson'].append(np.abs(simpson_approx - exact_integral))
    errors['Gauss3'].append(np.abs(gauss3_approx - exact_integral))

# 打印误差
for method, error_list in errors.items():
    print(f"{method} 误差: {error_list}")


