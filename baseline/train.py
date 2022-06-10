import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import os
import pandas as pd
import timm
import cv2
import torch.nn as nn
import torch.optim as optim
from dataset import BaseAugmentation, TrainDataset, ValidDataset
from torch.utils.data import Dataset, DataLoader
import torch.utils as utils
import mlflow
import mlflow.sklearn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from loss import F1Loss
import wandb
import copy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from loss import create_criterion, eval_criterion
from utils import Metric
import torch.nn.functional as F

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        os.makedirs(path, exist_ok=True)
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        os.makedirs(f'{path}{n}', exist_ok=True)
        return f"{path}{n}"


def train_load(key, train_path):

    train_images = []                                                       # 먼저 학습 데이터에 대한 이미지와 라벨을 모으기 위해 리스트 두 개를 선언한다.
    train_labels = []
    label_num_dict = {0:0, 1:0, 2:0, 3:0, 4:0}
    image_shapes = set()

    for file_name in os.listdir(train_path):
        if 'jpg' in file_name:
            json_name = file_name.replace('jpg', 'json')                       # 이미지에 대한 라벨링 json파일의 이름을 저장한다. (.jpg를 .json으로 변경하면 된다.)

            with open(os.path.join(train_path, json_name), "r") as json_file:   # json 파일에 접근하여 json 파일을 불러온다.
                img_json = json.load(json_file)             
        
            label = img_json[key]                                               # 학습하고자 하는 category의 라벨을 저장한다.

            if label < 0: continue                                              # 라벨이 -2, -1인 경우 학습에서 제외하여야 한다.

            label_num_dict[label] += 1

            image_path = os.path.join(train_path, file_name)                   # 이미지 경로를 불러온다.
            image = cv2.imread(image_path)                                      # 이미지를 불러오고 BGR을 RGB로 변경해준다.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_shapes.add(image.shape)
            train_images.append(image)
            train_labels.append(label)
    
    print('각 라벨 별 이미지 개수:', label_num_dict)

    return train_images, train_labels


def valid_load(key, valid_path):

    valid_images = []                                                       # valid셋에 대해서도 동일한 작업을 위해 두 개의 리스트를 선언한다.
    valid_labels = []
    label_num_dict_val = {0:0, 1:0, 2:0, 3:0, 4:0}
    image_shapes_val = set()

    for file_name in os.listdir(valid_path):
        if 'jpg' in file_name:
            json_name = file_name.replace('jpg', 'json')                       # 이미지에 대한 라벨링 json파일의 이름을 저장한다. (.jpg를 .json으로 변경하면 된다.)

            with open(os.path.join(valid_path, json_name), "r") as json_file:   # json 파일에 접근하여 json 파일을 불러온다.
                img_json = json.load(json_file)             

            label = img_json[key]                                               # 학습하고자 하는 category의 라벨을 저장한다.

            if label < 0: continue                                              # 라벨이 -2, -1인 경우 학습에서 제외하여야 한다.

            label_num_dict_val[label] += 1

            image_path = os.path.join(valid_path, file_name)                   # 이미지 경로를 불러온다.
            image = cv2.imread(image_path)                                      # 이미지를 불러오고 BGR을 RGB로 변경해준다.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_shapes_val.add(image.shape)
            valid_images.append(image)
            valid_labels.append(label)

    print('각 라벨 별 이미지 개수:', label_num_dict_val)

    return valid_images, valid_labels


def train(data_dir, model_dir, valid_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    print('pth 파일 저장 위치:', save_dir)
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- key
    key = args.key  # default: wrinkle

    # -- augmentation
    train_transform = BaseAugmentation(resize=args.resize)
    valid_transform = BaseAugmentation(resize=args.resize)

    # -- data_loader
    print("Train Image Loading...")
    train_images, train_labels = train_load(key, data_dir)
    print("Train Image Loading End & Valid Image Loading...")
    valid_images, valid_labels = valid_load(key, valid_dir)
    print("Valid Image Loading End")
    train_dataset = TrainDataset(train_images, train_labels, train_transform.transform)
    valid_dataset = ValidDataset(valid_images, valid_labels, valid_transform.transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    print("Model Loading...")
    model_module = getattr(import_module("model"), args.model)  # default: MyEfficientNet
    model = model_module(
        num_classes=5
    ).to(device)
    model = torch.nn.DataParallel(model)
    print("Model Loading End")
    

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        # weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)


    train_loss_list = []                                                # 이후 학습 그래프를 그리기 위해 선언한 리스트 들이다. loss값과 accuracy값을 각각 저장한다.
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    total_train_image = len(train_dataset)                              # 총 학습 이미지의 개수를 의미한다.
    total_train_batch = len(train_loader)                               # 각 에포크 당 미니 배치 개수를 의미한다.
    total_val_image = len(valid_dataset)    
    total_val_batch = len(valid_loader)


    best_acc = 0
    soft_mse_labels=[torch.tensor([[0.95, 0.05, 0, 0, 0]]),
        torch.tensor([[0.05, 0.9, 0.05, 0, 0]]),
        torch.tensor([[0, 0.05, 0.9, 0.05, 0]]),
        torch.tensor([[0, 0, 0.05, 0.9, 0.05]]),
        torch.tensor([[0, 0, 0, 0.05, 0.95]]),
        ]
    for epoch in range(args.epochs):                                    # args.epochs 값 만큼 epoch 실행한다.
        model.train()                                                     # model 학습 모드로 변경한다.
        train_accuracy = 0                                                # 해당 epoch의 accuracy와 loss를 저장할 변수 선언한다.
        train_loss = 0                        
        # train_precision, train_recall, train_f1 = 0, 0, 0
        metric = Metric()
        for images, labels in tqdm(train_loader):   
            images = images.to(device)
            labels = labels.to(device)
            hypothesis = model(images)               

            '''
             loss : 'cross_entropy','focal','label_smoothing' 'f1', 'CB' 
            '''              
            loss = criterion(hypothesis, labels)                            

            '''
            loss : 'softmax_ce'
            '''
            # mse_hypothesis = F.softmax(hypothesis, dim=1)
            # mse_labels=torch.tensor([])
            # for i in labels:
            #     mse_labels = torch.cat((mse_labels,soft_mse_labels[int(i)].clone().detach()),0)
            # mse_labels = mse_labels.to(device)
            # loss = criterion(mse_hypothesis, mse_labels) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = torch.argmax(hypothesis, 1)                        # 학습 이미지에 대해 모델이 예측한 label을 저장한다.
            correct = (prediction == labels)                                # 정답 label들과 비교한다.
            train_accuracy += correct.sum().item() / total_train_image      # accuracy 값을 갱신한다.
            train_loss += loss.item() / total_train_batch                   # loss 값을 갱신한다.
            
            metric.add_data(hypothesis, labels)

            # precision, recall, f1 = eval_criterion(hypothesis, labels)
            # train_precision += precision / total_train_batch
            # train_recall += recall / total_train_batch
            # train_f1 += f1 / total_train_batch
        
        train_recall = metric.get_recall()
        train_wrecall = metric.get_w_recall()
        train_precall = metric.get_p_recall()
            
            
        # wandb.log({
        #     "train_loss": train_loss,
        #     "train_accuracy": train_accuracy,
        #     "train_precision": train_precision,
        #     "train_recall": train_recall,
        #     "train_f1": train_f1
        # })   
        scheduler.step()

        val_accuracy = 0
        val_loss = 0
        # valid_precision, valid_recall, valid_f1 = 0, 0, 0

        with torch.no_grad():
            model.eval()
            metric = Metric()
            for images, labels in tqdm(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                prediction = model(images)
                
                '''
                loss : 'cross_entropy','focal','label_smoothing' 'f1', 'CB' 
                '''              
                loss = criterion(prediction, labels)
                      
                '''
                loss : 'softmax_ce'
                '''
                # mse_prediction = F.softmax(prediction, dim=1)
                # mse_labels=torch.tensor([])
                # for i in labels:
                #     mse_labels = torch.cat((mse_labels,soft_mse_labels[int(i)].clone().detach()),0)
                # mse_labels = mse_labels.to(device)
                # loss = criterion(mse_prediction, mse_labels) 

                correct = (torch.argmax(prediction, 1) == labels)
                val_accuracy += correct.sum().item() / total_val_image
                val_loss += loss.item() / total_val_batch

                metric.add_data(prediction, labels)
                
                # precision, recall, f1 = eval_criterion(prediction, labels)
                # valid_precision += precision / total_val_batch
                # valid_recall += recall / total_val_batch
                # valid_f1 += f1 / total_val_batch
            
            valid_recall = metric.get_recall()
            valid_wrecall = metric.get_w_recall()
            valid_precall = metric.get_p_recall()
                
            
            #베스트 모델 저장 
            if best_acc < val_accuracy:
                best_acc = val_accuracy
                torch.save(model.state_dict(), f'{save_dir}/{key}_best_acc_model.pth')
                # torch.save(model.module.state_dict(), f"{save_dir}/{key}_best_acc_module.pth")
                print('Model Saved.')
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                # best_model_wts = copy.deepcopy(model.state_dict())
                
                
            # wandb.log({
            #     "val_loss": val_loss,
            #     "val_accuracy": val_accuracy,
            #     "valid_recall": valid_recall,
            #     "valid_w_recall": valid_precall,
            #     "valid_p_recall": valid_wrecall
            # })
            
        print(f'[epoch {epoch+1}/{args.epochs}] train_loss: {train_loss:.5} train_accuracy: {train_accuracy:.5} val_loss: {val_loss:.5} val_accuracy: {val_accuracy:.5}')
        print(f'[epoch {epoch+1}/{args.epochs}] train_recall: {train_recall:.5} train_wrecall: {train_wrecall:.5} train_precall: {train_precall:.5}')
        print(f'[epoch {epoch+1}/{args.epochs}] valid_recall: {valid_recall:.5} valid_wrecall: {valid_wrecall:.5} valid_precall: {valid_precall:.5}')

        train_acc_list.append(train_accuracy)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_accuracy)
        val_loss_list.append(val_loss)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    # parser.add_argument('--dataset', type=str, default='MyEfficientNet', help='dataset augmentation type (default: MyEfficientNet)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[512, 512], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    # parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='MyEfficientNet', help='model type (default: MyEfficientNet)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    # parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--key', type=str, default='wrinkle', help='choose category(hydration, oil, pigmentation, sensitive, wrinkle)')

    # Container environment
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/naverboostcamp_dataset/naverboostcamp_train/JPEGImages'))
    parser.add_argument('--valid_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/naverboostcamp_dataset/naverboostcamp_val/JPEGImages'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.train_dir
    valid_dir = args.valid_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, valid_dir, args)