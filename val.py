from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from model.resnet import resnet18, resnet34
from model.price_model import Uniqlo
from model.category_model import Uniql_category_model
from model.name_model import Uniqlo_name_model
from model.price_model_classification import Uniqlo_price_cls_model
import os

def preprocess_image(image):
    #convert image to RGB
    image = image.convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize image to match model input size
        T.ToTensor()  # Convert PIL image to PyTorch tensor
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def get_price_from_index(index, csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Mapping sub-category to index
    price_mapping = {}
    for _, row in df.iterrows():
        sub_category = row['Price']
        idx = row['index']
        price_mapping[idx] = sub_category
    
    # Retrieve category from index
    if index in price_mapping:
        return price_mapping[index]
    else:
        return None
    
def predict_cls_price(image):
    backbone = resnet34()
    model = Uniqlo_price_cls_model(backbone)
    
    model_path = 'checkpoint/price_cls_overfit.pt'  # Path to the trained model
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        # max predict
        _, predicted = torch.max(prediction, 1)
        return get_price_from_index(predicted.item(), 'datasets/csv_for_price/image_data.csv')



def count_acc(csv_file, image_dir):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    correct = 0
    total = 0
    for _, row in df.iterrows():
        # print(int(row['Image Name']))
        image_path = os.path.join(image_dir, (str(int(row['Image Name'])) + '.jpg'))
        image = Image.open(image_path)
        image = preprocess_image(image)  # Assuming you have this function defined
        index = predict_cls_price(image)  # Assuming you have this function defined
        price = get_price_from_index(row['index'], 'datasets/csv_for_price/image_valid.csv')
        if price == row['Price']:
            correct += 1
        total += 1
    return correct / total

print(count_acc('datasets/csv_for_price/image_valid.csv', 'datasets/images'))
