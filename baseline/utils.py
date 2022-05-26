import torch
import torch.nn.functional as F

class Metric:
    def __init__(self):
        self.matrix = [[0]*5 for _ in range(5)]
        self.epsilon = 1e-7

    def add_data(self, preds, labels):
        size = len(labels)
        preds = torch.argmax(preds, dim=-1)
        for s in range(size):
            self.matrix[int(labels[s])][int(preds[s])] += 1

    def calculate(self):
        pre = self._precision()
        rec = self._recall()
        f1 = []
        for i in range(5):
            temp = 2 * (pre[i] * rec[i]) / (pre[i] + rec[i] + self.epsilon)
            f1.append(temp)
        
        return sum(pre)/5, sum(rec)/5, sum(f1)/5

    def _precision(self):
        res = []
        for i in range(5):
            temp = []
            for j in range(5):
                temp.append(self.matrix[j][i])
            res.append(temp[i] / (sum(temp) + self.epsilon))
        return res

    def _recall(self):
        res = []
        for i in range(5):
            temp = self.matrix[i][i] / (sum(self.matrix[i]) + self.epsilon)
            res.append(temp)
        return res