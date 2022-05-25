
import torch
from torchvision.transforms import ToTensor,ToPILImage
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


class GradcamModule:
    def __init__(self,grad_model,target_layer,height,width):
        self.model=grad_model
        self.target_layer=target_layer
        self.height=height
        self.width=width
        self.num_features=self.target_layer.num_features 
        
    def result(self,rgb_img):
        self.rgb_img = rgb_img

        def reshape_transform(tensor):
            value = tensor.reshape(tensor.size(0), self.height, self.width, self.num_features)
            value = value.transpose(2, 3).transpose(1, 2)

            return value

        self.target_layers_list = [self.target_layer]

        for param in self.model.parameters():
            param.requires_grad = True
        self.cam = GradCAMPlusPlus(model=self.model, 
                    target_layers=self.target_layers_list, 
                    reshape_transform=reshape_transform, 
                    use_cuda=torch.cuda.is_available())

        self.input_tensor = preprocess_image(self.rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.grayscale_cam = self.cam(input_tensor=self.input_tensor,
                                targets=None,)

        self.grayscale_cam = self.grayscale_cam[0, :]
        self.cam_image = show_cam_on_image(self.rgb_img, self.grayscale_cam, use_rgb=True)

        return self.cam_image

        