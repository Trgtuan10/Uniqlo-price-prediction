import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

# Step 1: Load the CSV file
csv_file = "UniqloFinal.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Step 2: Create a directory to store downloaded images
image_dir = "datasets/imagesss"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Step 3: Download images and preprocess data
image_paths = []
prices = []

for index, row in data.iterrows():
    img_url = row['Images-src']  # Assuming 'img_url' is the column name containing image URLs
    price = row['Price']      # Assuming 'price' is the column name containing prices

    # Remove currency symbol "VND" and convert string to integer
    price = int(price.replace("VND", "").replace(".", ""))

    # Download image
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img_path = os.path.join(image_dir, f"image_{index}.jpg")  # Save each image with a unique name
        img.save(img_path)
        image_paths.append(img_path)
        prices.append(price)
        print(f"Downloaded image {index + 1}/{len(data)}")
    except Exception as e:
        print(f"Error downloading image {index + 1}: {e}")

# Step 4: Create DataFrame with image paths and prices
image_data = pd.DataFrame({'image_path': image_paths, 'price': prices})

# Step 5: Save the image data to a new CSV file
image_data.to_csv("image_data.csv", index=False)

print("Image data saved successfully!")
