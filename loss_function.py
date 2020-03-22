import numpy as np
class LossFunction:
    # SoftmaxWithLoss函数及其导数的定义
    def softmax_logloss(self, inputs, label):
        temp1 = np.exp(inputs)
        probability = temp1 / (np.tile(np.sum(temp1, 1), (inputs.shape[1], 1))).T
        temp3 = np.argmax(label, 1)  # 纵坐标
        temp4 = [probability[i, j] for (i, j) in zip(np.arange(label.shape[0]), temp3)]
        loss = -1 * np.mean(np.log(temp4))
        return loss

    def der_softmax_logloss(self, inputs, label):
        temp1 = np.exp(inputs)
        temp2 = np.sum(temp1, 1)  # 它得到的是一维的向量;
        probability = temp1 / (np.tile(temp2, (inputs.shape[1], 1))).T
        gradient = probability - label
        return gradient

    def least_square_loss(self, predict, label):
        tmp1 = np.sum(np.square(label - predict), 1)
        loss = np.mean(tmp1)
        return loss

    def der_least_square_loss(self, predict, label):
        gradient = predict - label
        return gradient