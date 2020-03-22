import numpy as np
class MetricCalculator:
    def __init__(self, label, predict):
        self.label = label
        self.predict = predict
        assert len(label)==len(predict), "length of label and predict must be equal"
        self.mse = None
        self.rmse = None
        self.mae = None
        self.auc = None

    def get_mse(self):
        self.mse = np.mean(np.sum(np.square(self.label - self.predict),1))

    def get_rmse(self):
        self.rmse = np.sqrt(np.mean(np.sum(np.square(self.label - self.predict), 1)))

    def get_mae(self):
        self.mae = np.mean(np.sum(np.abs(self.label - self.predict),1))

    def get_auc(self):
        prob = self.predict.reshape(-1).tolist()
        label = self.label.reshape(-1).tolist()
        f = list(zip(prob, label))
        rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
        posNum = 0
        negNum = 0
        for i in range(len(label)):
            if (label[i] == 1):
                posNum += 1
            else:
                negNum += 1
        self.auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)

    def print_metrics(self):
        if(self.mse): print("mse: ",self.mse)
        if(self.rmse): print("rmse: ",self.rmse)
        if(self.mae): print("mae: ",self.mae)
        if(self.auc): print("auc: ",self.auc)
