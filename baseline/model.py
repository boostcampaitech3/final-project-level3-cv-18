import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import timm


class MyEfficientNet(nn.Module) :
    '''
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 5개의 Class를 예측하는 형태의 Model입니다.
    '''
    def __init__(self, num_classes: int = 5) :
        super(MyEfficientNet, self).__init__()
        self.EFF = EfficientNet.from_pretrained('efficientnet-b4', in_channels=3, num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.EFF(x)
        x = F.softmax(x, dim=1)
        return x


class MyEfficientNetTimm(nn.Module) :
    '''
    timm을 사용한 모델입니다.
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 5개의 Class를 예측하는 형태의 Model입니다.
    '''
    def __init__(self, num_classes: int = 5) :
        super(MyEfficientNetTimm, self).__init__()
        self.EFF = timm.create_model('efficientnet_b4', pretrained=True, num_classes=5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.EFF(x)
        x = F.softmax(x, dim=1)
        return x