import torch
import torch.nn.functional as F

class Metric:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.matrix = [[0]*self.num_classes for _ in range(self.num_classes)]
        self.weights = [0]*(self.num_classes - 2) + [0.5, 1, 0.5] + [0]*(self.num_classes - 2)
        self.penalty = [-0.5 * i for i in range(self.num_classes - 2, 0, -1)] + [0.5, 1, 0.5] + [-0.5 * i for i in range(1, self.num_classes - 1)]
        self.epsilon = 1e-7

    def add_data(self, preds, labels):
        size = len(labels)
        preds = torch.argmax(preds, dim=-1)
        for s in range(size):
            self.matrix[int(labels[s])][int(preds[s])] += 1

    def _precision(self):
        res = []
        for i in range(self.num_classes):
            temp = []
            for j in range(self.num_classes):
                temp.append(self.matrix[j][i])
            res.append(temp[i] / (sum(temp) + self.epsilon))
        return res

    def _recall(self):
        res = []
        for i in range(self.num_classes):
            temp = self.matrix[i][i] / (sum(self.matrix[i]) + self.epsilon)
            res.append(temp)
        return res

    def _w_recall(self):
        res = []
        for i in range(self.num_classes):
            w = self.weights[self.num_classes - 1 - i: self.num_classes*2 - 1 - i]
            temp = 0
            for j in range(self.num_classes):
                temp += self.matrix[i][j] * w[j]
            temp /= (sum(self.matrix[i]) + self.epsilon)
            res.append(temp)
        return res

    def _p_recall(self):
        res = []
        for i in range(self.num_classes):
            p = self.penalty[self.num_classes - 1 - i: self.num_classes*2 - 1 - i]
            temp = 0
            for j in range(self.num_classes):
                temp += self.matrix[i][j] * p[j]
            temp /= (sum(self.matrix[i]) + self.epsilon)
            res.append(temp)
        return res

    def get_precision(self):
        pre = self._precision()
        return sum(pre)/len(pre)

    def get_recall(self):
        rec = self._recall()
        return sum(rec)/len(rec)

    def get_w_recall(self):
        wrec = self._w_recall()
        return sum(wrec)/len(wrec)

    def get_p_recall(self):
        prec = self._p_recall()
        return sum(prec)/len(prec)


def get_score(outputs):
    p = torch.nn.Softmax(dim=-1)(outputs)
    preds = p * torch.Tensor([0, 1, 2, 3, 4])
    return torch.sum(preds, dim=-1)

def MAE(outputs, labels):
    scores = get_score(outputs)
    mae = torch.abs(preds - labels)

    return torch.mean(mae)

def MSE(outputs, labels):
    scores = get_score(outputs)
    mse = preds - labels
    mse *= mse

    return torch.mean(mse)