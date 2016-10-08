# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def numeric_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.plot(x, y)
# plt.show()

y = numeric_diff(function_1, x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()