import os
import json

from tqdm import tqdm
from glob import glob
import shutil

import cv2
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

# criterion에 들어가는 값들을 그대로 넣으시면 됩니다.
def MAE(outputs, labels):
    scores = get_score(outputs)
    mae = torch.abs(preds - labels)

    return torch.mean(mae)

def MSE(outputs, labels):
    scores = get_score(outputs)
    mse = preds - labels
    mse *= mse

    return torch.mean(mse)

class LipsCrop:
    def make_crop_image(data_path,save_path):
        if not os.path.exists(save_path):
                os.makedirs(save_path)

        # 턱 이미지만 골라오기
        chin_image_path = []

        annotation_paths = glob(os.path.join(data_path,'*.json'))
        for path in annotation_paths:
            with open(path) as f:
                anno = json.load(f)

                # 턱 이미지가 아니라면 무시
                if anno['part'] != 3:
                    continue
            
                chin_image_path.append(os.path.join(data_path,anno['file_name']))
                shutil.copyfile(path, os.path.join(save_path, '1' + anno['file_name'].split('.')[0]+'.json'))

        chin_image_path.sort()
        
        # 입술을 검출해주는 모델 생성
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        # 턱 사진에서 입술만 크롭하여 저장하기
        for path in tqdm(chin_image_path):
            img = cv2.imread(path)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            file_name = '1' + path.split('/')[-1]

            # 입술 ROI 영역 생성
            mouths = mouth_cascade.detectMultiScale(gray,1.1,20)

            # 후보 영역이 여러개인 경우
            mouth_list = []
            if len(mouths) != 1:
                for (x,y,w,h) in mouths:
                    mouth_list.append(w)
                    
                if len(mouth_list) == 0:
                    continue

                else:
                    idx = mouth_list.index(max(mouth_list))
                    mouth = mouths[idx]
            
            else:
                mouth = mouths[0]
            
            x,y,w,h = mouth
            
            crop_image = img[y:y+h,x:x+w]
            cv2.imwrite(os.path.join(save_path,file_name),crop_image)