import numpy as np

# 定义函数 f(x) = e^x
def f(x):
    return np.exp(x)

# 定义导数的近似公式
def approximate_derivative(f, x, h):
    return (f(x-2*h) - 4*f(x-h) + 3*f(x)) / (2*h)

# 精确导数 f'(x) = e^x
def exact_derivative(x):
    return np.exp(x)

# 在 x = 0 处计算导数
x = 0
h_values = [10**-i for i in range(1, 9)]
errors = []

for h in h_values:
    approx_derivative = approximate_derivative(f, x, h)
    exact_derivative_value = exact_derivative(x)
    error = np.abs(approx_derivative - exact_derivative_value)
    errors.append(error)
    print(f"h = {h}, 近似导数 = {approx_derivative}, 精确导数 = {exact_derivative_value}, 误差 = {error}")

# 打印误差
print("误差列表:", errors)




