from fastapi import FastAPI, File, UploadFile
from prediction import predict
import uvicorn
import numpy as np
from typing import List
import cv2
from image_crop import crop
from grad_cam import GradCam,show_cam_on_image,preprocess_image
from efficientnet_pytorch import EfficientNet
from fastapi import FastAPI
import torch
import time

app = FastAPI()

model_path = ['/opt/ml/input/artlab/models/wrinkle_best_precall.pth','/opt/ml/input/artlab/models/oil.pth','/opt/ml/input/artlab/models/sensitive (1).pth','/opt/ml/input/artlab/results/b0_pigmentation_softlabel_best.pt','/opt/ml/input/artlab/results/hydration_best_precall_model.pth']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model0 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
model0.load_state_dict(torch.load(model_path[0],map_location=device))
model1 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
model1.load_state_dict(torch.load(model_path[1],map_location=device))
model2 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
model2.load_state_dict(torch.load(model_path[2],map_location=device))
model3 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
model3.load_state_dict(torch.load(model_path[3],map_location=device))
model4 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
model4.load_state_dict(torch.load(model_path[4],map_location=device))


@app.post('/gradcam')
async def get_gradcam(files: List[UploadFile] = File(...)):
    global model0,model1,model2,model3,model4,model_path,device
    image_bytes = await files[0].read()
    encoded_img = np.frombuffer(image_bytes,dtype=np.uint8)
    new_img = cv2.imdecode(encoded_img,cv2.IMREAD_COLOR)

    crop_img = crop(new_img)
    cam_list = []
    label_list = []
    start= time.time()
    for i in range(5):
        if i == 0:
            model = model0
        elif i == 1:
            model = model1
        elif i == 2:
            model = model2
        elif i == 3:
            model = model3
        else:
            model = model4
        #model.load_state_dict(torch.load(model_path[i],map_location=device))
        label = predict(model=model,img=crop_img)
        label_list.append(label)
        model = model.to('cpu')  
        rgb_img=cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
        rgb_img = np.float32(rgb_img) / 255

        cam_img = preprocess_image(rgb_img)
        # target layer
        target_layer_names = [str(len(model._blocks) - 1)]
        img_size = (rgb_img.shape[1], rgb_img.shape[0])
        grad_cam = GradCam(model=model, blob_name = '_blocks', target_layer_names=target_layer_names, use_cuda=True,img_size=img_size)
        target_index = label
        mask_dic = grad_cam(cam_img, target_index)
        for k,v in mask_dic.items():
            gray_mask=v
            name=k
        cam_image = show_cam_on_image(rgb_img, 1-gray_mask, name)
        cam_list.append(cam_image.tolist())
    end = time.time()
    print(end-start)
    return {'cam':cam_list,'label':label_list}

def main():
    model_path = ['/opt/ml/input/artlab/results/wrinkle_best_model (1) (1).pth','/opt/ml/input/artlab/models/oil.pth','/opt/ml/input/artlab/models/sensitive (1).pth','/opt/ml/input/artlab/results/b0_pigmentation_softlabel_best.pt','/opt/ml/input/artlab/results/hydration_best_precall_model.pth']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model0 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model0.load_state_dict(torch.load(model_path[0],map_location=device))
    model1 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model1.load_state_dict(torch.load(model_path[1],map_location=device))
    model2 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model2.load_state_dict(torch.load(model_path[2],map_location=device))
    model3 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model3.load_state_dict(torch.load(model_path[3],map_location=device))
    model4 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model4.load_state_dict(torch.load(model_path[4],map_location=device))
#    uvicorn.run(app, host="172.17.0.2", port=30002, workers=5)

if __name__ == '__main__':
    model_path = ['/opt/ml/input/artlab/results/wrinkle_best_model (1) (1).pth','/opt/ml/input/artlab/models/oil.pth','/opt/ml/input/artlab/models/sensitive (1).pth','/opt/ml/input/artlab/results/b0_pigmentation_softlabel_best.pt','/opt/ml/input/artlab/results/hydration_best_precall_model.pth']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model0 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model0.load_state_dict(torch.load(model_path[0],map_location=device))
    model1 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model1.load_state_dict(torch.load(model_path[1],map_location=device))
    model2 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model2.load_state_dict(torch.load(model_path[2],map_location=device))
    model3 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model3.load_state_dict(torch.load(model_path[3],map_location=device))
    model4 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5).to(device)
    model4.load_state_dict(torch.load(model_path[4],map_location=device))
    #main()


