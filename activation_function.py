import numpy as np
class ActivationFunction:
    def __init__(self):
        pass
    # sigmoid函数及其导数的定义
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def der_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # tanh函数及其导数的定义
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def der_tanh(self, x):
        return 1 - self.tanh(x) * self.tanh(x)

    # ReLU函数及其导数的定义
    def relu(self, x):
        temp = np.zeros_like(x)
        if_bigger_zero = (x > temp)
        return x * if_bigger_zero

    def der_relu(self, x):
        temp = np.zeros_like(x)
        if_bigger_equal_zero = (x >= temp)  # 在零处的导数设为1
        return if_bigger_equal_zero * np.ones_like(x)

    # Identity函数及其导数的定义
    def identity(self, x):
        return x

    def der_identity(self, x):
        return x