import streamlit as st
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

# Function to preprocess the uploaded image
def preprocess_image(image):
    #convert image to RGB
    image = image.convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize image to match model input size
        T.ToTensor()  # Convert PIL image to PyTorch tensor
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict price based on the image
def predict_price(image):
    backbone = resnet18()
    model = Uniqlo(backbone)

    model_path = 'checkpoint/model_state_dict.pt'  # Path to the trained model
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    model.load_state_dict(checkpoint)
    
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        return prediction.item()
    
    
    
def get_category_from_index(index, csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Mapping sub-category to index
    sub_category_mapping = {}
    for _, row in df.iterrows():
        sub_category = row['sub-category']
        idx = row['index']
        sub_category_mapping[idx] = sub_category
    
    # Retrieve category from index
    if index in sub_category_mapping:
        return sub_category_mapping[index]
    else:
        return None
    
def get_name_from_index(index, csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Mapping sub-category to index
    name_mapping = {}
    for _, row in df.iterrows():
        sub_category = row['Title']
        idx = row['index']
        name_mapping[idx] = sub_category
    
    # Retrieve category from index
    if index in name_mapping:
        return name_mapping[index]
    else:
        return None

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
    
def predict_category(image):
    backbone = resnet18()
    model = Uniql_category_model(backbone)

    model_path = 'checkpoint/best_model_cate.pt'  # Path to the trained model
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        # max predict
        _, predicted = torch.max(prediction, 1)
        return get_category_from_index(predicted.item(), 'datasets/csv_for_category/data_cate.csv')
        
def predict_name(image):
    backbone = resnet34()
    model = Uniqlo_name_model(backbone)

    model_path = 'checkpoint/name_model.pt'  # Path to the trained model
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        # max predict
        _, predicted = torch.max(prediction, 1)
        return get_name_from_index(predicted.item(), 'datasets/csv_for_name/data_name.csv')


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

# Streamlit App
def main():
    st.title("Uniqlo Information Predictor")
    st.write("Upload an image and we'll predict its price, category and title!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Button to trigger prediction
        if st.button('Predict'):
            # Predict price
            # price_prediction = predict_price(processed_image)       #regression
            
            price_prediction = predict_cls_price(processed_image)       #cls
            category_prediction = predict_category(processed_image)
            name_prediction = predict_name(processed_image)
            
            # st.write(f"Predicted Price:    {round(price_prediction / 1000) * 1000:,} VND") #regression
            
            st.write(f"Predicted Price:  {int(price_prediction * 100000):,} VND")    #cls
            st.write(f"Predicted Category:    {category_prediction}")
            st.write(f"Predicted Name:    {name_prediction}")



# Run the app
if __name__ == "__main__":
    main()
