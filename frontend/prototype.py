import streamlit as st
from PIL import Image
import io
import numpy as np
import torch
import cv2
st.title("피부 평가 (CV 18조)")
import requests
from image_crop import crop


torch.cuda.empty_cache()
img_file = st.camera_input('')

global crop_img

if img_file : 
    image_bytes = img_file.getvalue()
    img =Image.open(io.BytesIO(image_bytes))
    encoded_img = np.frombuffer(img_file.getvalue(),dtype=np.uint8)
    new_img = cv2.imdecode(encoded_img,cv2.IMREAD_COLOR)
    try:
        crop(new_img)
        crop_img = cv2.cvtColor(crop(new_img),cv2.COLOR_BGR2RGB)
        files = [
            ('files', (img_file.name, image_bytes,
                       img_file.type))
        ]
        cam_response = requests.post("http://101.101.217.13:30002/gradcam", files=files)
        cam_result = (np.array(cam_response.json()['cam']))
        label = cam_response.json()['label']
        st.image(cam_result,caption=f"label : {label}")
    except:
        st.write('사진을 다시 찍어주세요.')



   

