# -*- coding: utf-8 -*-
from made_5_1_multioplelayer import MulLayer

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dout, dout

if __name__ == '__main__':
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    # forward
    mal_apple_layer = MulLayer()
    mal_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    apple_price = mal_apple_layer.forward(apple, apple_num)
    orange_price = mal_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    print(price)

    # backward
    dout = 1
    dall_price, dtax = mul_tax_layer.backward(dout)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dapple, dapple_num = mal_apple_layer.backward(dapple_price)
    dorange, dorange_num = mal_orange_layer.backward(dorange_price)
    print(dapple_num, dapple, dorange, dorange_num, dtax)
