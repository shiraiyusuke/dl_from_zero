# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.pardir)
import numpy as np


class simpleNet():
    def __init__(self):
        self.W = np.random.randn(2, 3) # 正規分布でWを初期化

    def predict(self, x):
            return np.dot(x, self.W)

    def loss(self, x, t):
            z = self.predict(x)
            y = softmax(z)
            loss = closs_entropy_error(y, t)
            return loss

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x.max(x, axis=1)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def closs_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

        # tがone hot vectorの場合、正解ラベルのインデックス番号へ変換
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext()

    return grad

net = simpleNet()
# print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
t = np.array([0, 0, 1])
print(net.loss(x, t))

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)