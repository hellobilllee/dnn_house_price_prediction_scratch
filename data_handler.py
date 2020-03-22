import numpy as np
import random
class DataHander:
    def __init__(self,batch_size):
        self.data_sample = 0
        self.data_label = 0
        self.output_sample = 0
        self.output_label = 0
        self.point = 0  # 用于记住下一次pull数据的地方;
        self.batch_size = batch_size

    def get_data(self, sample, label):  # sample 每一行表示一个样本数据, label的每一行表示一个样本的标签.
        self.data_sample = sample
        self.data_label = label

    def shuffle(self):  # 用于打乱顺序;
        random_sequence = random.sample(range(self.data_sample.shape[0]), self.data_sample.shape[0])
        self.data_sample = self.data_sample[random_sequence]
        self.data_label = self.data_label[random_sequence]

    def pull_data(self):  # 把数据推向输出
        start = self.point
        end = start + self.batch_size
        output_index = np.arange(start, end)
        if end > self.data_sample.shape[0]:
            end = end - self.data_sample.shape[0]
            output_index = np.append(np.arange(start, self.data_sample.shape[0]), np.arange(0, end))
        self.output_sample = self.data_sample[output_index]
        self.output_label = self.data_label[output_index]
        self.point = end % self.data_sample.shape[0]