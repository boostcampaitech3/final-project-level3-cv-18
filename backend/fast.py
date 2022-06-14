from fastapi import FastAPI, File, UploadFile
from prediction import predict
import uvicorn
import numpy as np
import cv2
from image_crop import crop
from grad_cam import GradCam,show_cam_on_image,preprocess_image
from efficientnet_pytorch import EfficientNet
from fastapi import FastAPI
import torch
import time

app = FastAPI()


@app.post('/gradcam')
async def get_gradcam(files: UploadFile = File(...)):
    global model
    image_bytes = await files.read() # file 읽어오기
    encoded_img = np.frombuffer(image_bytes,dtype=np.uint8) # image crop 하기 위한 과정
    new_img = cv2.imdecode(encoded_img,cv2.IMREAD_COLOR)

    crop_img = crop(new_img)
    label = predict(model=model,img=crop_img) # label 값

    rgb_img=cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR) # grad-cam 출력 과정
    rgb_img = np.float32(rgb_img) / 255

    cam_img = preprocess_image(rgb_img)
    # target layer
    target_layer_names = [str(len(model._blocks) - 1)]
    img_size = (rgb_img.shape[1], rgb_img.shape[0])
    grad_cam = GradCam(model=model, blob_name = '_blocks', target_layer_names=target_layer_names, use_cuda=True,img_size=img_size)
    mask_dic = grad_cam(cam_img, label)

    for k,v in mask_dic.items():
        gray_mask=v
        name=k

    cam_image = show_cam_on_image(rgb_img, 1-gray_mask, name)
    cam_list = cam_image.tolist()
    #cam_list.append(cam_image.tolist())
    return {'cam':cam_list,'label':label}




def main():
    uvicorn.run(app, host="172.17.0.2", port=30002)

if __name__ == '__main__':
    model_path = '/opt/ml/input/artlab/models/wrinkle_best_precall.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    main()
