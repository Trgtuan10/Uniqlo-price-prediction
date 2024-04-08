import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from resnet import resnet18
from Uniqlo import Uniqlo

# Function to preprocess the uploaded image
def preprocess_image(image):
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize image to match model input size
        T.ToTensor()  # Convert PIL image to PyTorch tensor
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict price based on the image
def predict_price(image):
    backbone = resnet18()
    model = Uniqlo(backbone)

    model_path = 'model_state_dict.pt'  # Path to the trained model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint)
    
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        return prediction.item()

# Streamlit App
def main():
    st.title("Uniqlo Price Predictor")
    st.write("Upload an image and we'll predict its price!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Button to trigger prediction
        if st.button('Predict Price'):
            # Predict price
            price_prediction = predict_price(processed_image)
            st.write(f"Predicted Price Range: {int(price_prediction - 50000) } VND - {int(price_prediction + 50000)} VND")


# Run the app
if __name__ == "__main__":
    main()
