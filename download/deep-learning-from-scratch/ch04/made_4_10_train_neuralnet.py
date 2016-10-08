# -*- coding: utf-8 -*-
import os, sys
import numpy as np
sys.path.append(os.pardir)
from made_4_8_twolayernet import TwoLayerNet
from dataset.mnist import load_mnist
import time



if __name__ == '__main__':
    start = time.time()


    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_loss_list = []
    train_acc_list =[]
    test_acc_list = []

    # ハイパーパラメータ
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    # 1epochあたりの繰り返し数
    iter_per_epoch = max(train_size / batch_size, 1)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    # ミニバッチの実装
    for i in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        # grad = network.numerical_gradient(x_batch, t_batch) #こっちは、数値微分での学習
        grad = network.gradient(x_batch, t_batch)             #こっちはSGD。むちゃくちゃ早い。。。

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
    elapsed_time = time.time() - start
    print (("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
