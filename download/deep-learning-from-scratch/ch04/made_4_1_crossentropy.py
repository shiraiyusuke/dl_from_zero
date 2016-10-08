# -*- coding: utf-8 -*-
import numpy as np

def cross_entropy(y, t):
    delta = 1e-7 # オーバーフロー対策
    return -np.sum(t * np.log(y + delta))


def test(y, t):
    print(cross_entropy(y, t))


t = [0  , 0   , 1  , 0  , 0   , 0  , 0  , 0  , 0  , 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(np.array(y).shape)
print(np.array(t).shape)
test(np.array(y), np.array(t))
