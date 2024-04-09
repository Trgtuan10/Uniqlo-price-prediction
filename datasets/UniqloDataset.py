import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class UniqloDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_column, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, str(self.annotations.iloc[index, 0]) + '.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # Trích xuất nhãn từ cột được chỉ định bằng tên
        y_label = torch.tensor(int(self.annotations.loc[index, self.label_column]))
        
        if self.transform:
            image = self.transform(image)
            
        return (image, y_label)
    

def get_transform(args):
    height = args.height
    width = args.width
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(),  
        T.RandomVerticalFlip(),    
        T.RandomRotation(degrees=30),  
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  
        T.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0)),
        T.ToTensor(),  # Chuyển ảnh sang tensor
])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
    ])

    return train_transform, valid_transform

