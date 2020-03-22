import numpy as np
class Optimizer:
    def __init__(self, lr = 0.01, momentum = 0.9, iteration = -1, gamma=0.0005, power=0.75):
        self.lr = lr
        self.momentum = momentum
        self.iteration = iteration
        self.gamma = gamma
        self.power = power
    # 固定方法
    def fixed(self):
        return self.lr

    # inv方法
    def anneling(self):
        if self.iteration == -1:
            assert False, '需要在训练过程中,改变update_method 模块里的 iteration 的值'
        self.lr = self.lr * np.power((1 + self.gamma * self.iteration), -self.power)
        return self.lr

    # 基于批量的随机梯度下降法
    def batch_gradient_descent_fixed(self, weights, grad_weights, previous_direction):
        direction = self.momentum * previous_direction + self.lr * grad_weights
        weights_now = weights - direction
        return (weights_now, direction)

    def batch_gradient_descent_anneling(self, weights, grad_weights, previous_direction):
        self.lr = self.anneling()
        direction = self.momentum * previous_direction + self.lr * grad_weights
        weights_now = weights - direction
        return (weights_now, direction)

    def update_iteration(self, iteration):
        self.iteration = iteration