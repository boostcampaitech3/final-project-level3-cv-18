from typing import Tuple

import streamlit as st
import torch
import yaml

from baseline.model import MyEfficientNet, MyEfficientNetTimm
from utils import transform_image


@st.cache
def load_model() -> MyEfficientNetTimm:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MyEfficientNet(num_classes=5).to(device)
    model = MyEfficientNetTimm(num_classes=5).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))

    return model


def get_prediction(model: MyEfficientNetTimm, image_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = transform_image(image_bytes=image_bytes).to(device)
    outputs = model.forward(tensor)

    _, y_hat = outputs.max(1)
    return tensor, y_hat
