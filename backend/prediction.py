import timm
import torch
from torchvision.transforms import ToTensor
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch
import torch.nn as nn


class MyEfficientNet(nn.Module):
    """
    EfiicientNet-b0의 출력층만 변경합니다.
    한번에 5개의 Class를 예측하는 형태의 Model입니다.
    """

    def __init__(self, num_classes: int = 5):
        super(MyEfficientNet, self).__init__()
        self.EFF = EfficientNet.from_pretrained("efficientnet-b0", in_channels=3, num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        x = self.EFF(x)
        x = F.softmax(x, dim=1)
        return x

def predict(model: MyEfficientNet,img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = ToTensor()(img)
    c,w,h = img.shape
    img = img.reshape(1,c,w,h).to(device)
    result = model(img)
    result = int(torch.argmax(result,1))
    return result

