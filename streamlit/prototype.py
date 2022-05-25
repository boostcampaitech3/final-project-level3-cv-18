
import streamlit as st
from PIL import Image
import io
import numpy as np
import timm
import torch.nn as nn
import torch
from torchvision.transforms import ToTensor
import cv2
from gradcam import GradcamModule

torch.cuda.empty_cache()
img_file = st.file_uploader("choose a file",type=["jpg","jpeg","png"])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model 종류 및 trained model 경로 설정
model = timm.create_model('tf_efficientnet_b5', pretrained=True, num_classes=5).to(device)
model.load_state_dict(torch.load('/opt/ml/project/trained_model/sensitive_effi_b5.pth'))
for param in model.parameters():
    param.requires_grad = True

# model에 따라 classifier,head 등 이름이 다르므로 수정 필요할 수 있음
in_features = model.classifier.in_features
model.head = nn.Sequential(
    nn.Linear(in_features, 5)
).to(device)
model.eval()


if img_file :
    img =Image.open(io.BytesIO(img_file.getvalue()))

    # 원본 사진 출력
    print_img = np.array(img)
    origin_h=print_img.shape[0]
    origin_w=print_img.shape[1]
    st.image(cv2.resize(cv2.cvtColor(print_img, cv2.COLOR_RGB2BGR), (origin_w//3, origin_h//3))[:, :, ::-1],caption="uploaded image!")

    # 모델 결과 출력
    input_img= img.convert('RGB')
    input_img = ToTensor()(input_img)
    c,w,h = input_img.shape
    input_img = input_img.reshape(1,c,w,h).to(device)

    result=model(input_img)


    # CAM 출력
    rgb_img=Image.open(io.BytesIO(img_file.getvalue()))
    rgb_img=np.array(rgb_img)  
    rgb_img=cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    rgb_img = cv2.resize(rgb_img, (512, 512))[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255

        # target layer
    target_layer = model.blocks[-1][-1].bn2

        # target layer feature 수 설정
    f_num_features = target_layer.num_features

        # gradcam_module.result 출력 이미지 크기
        # 단, num_features*height*width가 rgb_img의 size와 같아야함
    f_height = 16
    f_width = rgb_img.size//f_height//f_num_features

    gradcam_module=GradcamModule(model,target_layer,f_height,f_width)
    cam_img = gradcam_module.result(rgb_img)

    st.image(cv2.resize(cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR), (origin_w//3, origin_h//3))[:, :, ::-1],caption="trouble image")

