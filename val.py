import torch
import torchvision.transforms as T
from model.resnet import resnet50, resnet101, resnext50_32x4d,resnet152,resnet18,resnet34
from model.price_model import *

from PIL import Image


def test_price():
    backbone = resnet18()
    model = Uniqlo(backbone)

    model_path = 'v1.pt'  # luu dia chi cua model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['state_dict'])


    # Define the transformation pipeline
    transform = T.Compose([
        T.ToTensor()  # Convert PIL image to PyTorch tensor
    ])

    # Load the image
    img_path = 'datasets/images/image_2561.jpg'
    img = Image.open(img_path)

    # Apply the transformation
    img_tensor = transform(img)

    model.eval()
    print(model(img_tensor.unsqueeze(0)))
