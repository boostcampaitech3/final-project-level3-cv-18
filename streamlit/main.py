import streamlit as st
from PIL import Image
import io
import numpy as np
import timm
import torch.nn as nn
import torch
from torchvision.transforms import ToTensor
import cv2
from grad_cam import GradCam, GuidedBackpropReLUModel, show_cam_on_image, show_gbs, preprocess_image
import yaml
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt


st.title("피부 평가 (CV 18조)")
with open("config_wrinkle.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


torch.cuda.empty_cache()
img_file = st.file_uploader("choose a file",type=["jpg","jpeg","png"])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model 종류 및 trained model 경로 설정
# model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=5).to(device)
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)
model.load_state_dict(torch.load(config["model_path"]))
for param in model.parameters():
    param.requires_grad = True

# model에 따라 classifier,head 등 이름이 다르므로 수정 필요할 수 있음
# in_features = model.classifier.in_features
# model.head = nn.Sequential(
#     nn.Linear(in_features, 5)
# ).to(device)
model.eval()


if img_file :
    img =Image.open(io.BytesIO(img_file.getvalue()))
    # 원본 사진 출력
    print_img = np.array(img)
    origin_h=print_img.shape[0]
    origin_w=print_img.shape[1]
    st.image(cv2.resize(cv2.cvtColor(print_img, cv2.COLOR_RGB2BGR), (origin_w//3, origin_h//3))[:, :, ::-1],caption="uploaded image!")


    numpy_image=np.array(img) 
    input_img=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        area = (x,y-50,x+w,y+h+50)
        # cropped_img = input_img[y: y + h, x: x + w]
        cropped_img = img.crop(area)
        print(cropped_img.size)
    img = cropped_img.resize((512, 512))

    # crop 사진 출력
    print_img = np.array(img)
    origin_h=print_img.shape[0]
    origin_w=print_img.shape[1]
    st.image(cv2.resize(cv2.cvtColor(print_img, cv2.COLOR_RGB2BGR), (origin_w//3, origin_h//3))[:, :, ::-1],caption="Crop Image")

    target_layer_names = [str(len(model._blocks) - 1)]
    grad_cam = GradCam(model=model, blob_name = '_blocks', target_layer_names=target_layer_names, use_cuda=False)
    input_img= img.convert('RGB')
    input_img= np.array(input_img)
    img = np.float32(cv2.resize(input_img, (512, 512))) / 255
    inputs = preprocess_image(img)
    result = model(inputs)


    target_index = None
    mask_dic = grad_cam(inputs, target_index)
    for i, (name, mask) in enumerate(mask_dic.items()):
        cam_img = show_cam_on_image(img, mask, name)

    st.image(cv2.resize(cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR), (origin_w//3, origin_h//3))[:, :, ::-1],caption="Wrinkle CAM")
    label_num = torch.argmax(result, dim=-1)[0].item()
    label = config["classes"][label_num]
    col1, col2 = st.columns(2)
    col1.metric("Label", label[0])
    col2.metric("Description", label[1])
    print('Completed')
