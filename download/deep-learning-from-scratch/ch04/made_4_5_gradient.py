# -*- coding: utf-8 -*-
import numpy as np


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    """勾配(各要素の偏微分のベクトル計算)"""
    h = 1e-4
    grad = np.zeros_like(x) # xと同じ形状の配列を生成
    for idx in range(x.size): # xの要素数の数のループ
        tmp_val = x[idx]
        x[idx] = tmp_val + h # x+hを計算
        fxh1 = f(x)

        x[idx] = tmp_val - h # x-hを計算
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h) # 微分を計算
        x[idx] = tmp_val # 値を元に戻す

    return grad


print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))


