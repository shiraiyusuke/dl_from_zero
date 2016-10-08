# -*- coding: utf-8 -*-
import os, sys
import numpy as np
sys.path.append(os.pardir)
from made_4_8_twolayernet import TwoLayerNet
from dataset.mnist import load_mnist


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_loss_list = []

    # ハイパーパラメータ
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    # ミニバッチの実装
    for i in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        grad = network.numerical_gradient(x_batch, t_batch)
        # grad = network.gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        print(loss)
        train_loss_list.append(loss)
