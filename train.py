import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision

from datasets.UniqloDataset import UniqloDataset

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
batch_size = 16


#load data
dataset = UniqloDataset(csv_file='datasets/image_data.csv', 
                        root_dir='datasets/images', 
                        transform=transforms.ToTensor())

len_train = int(len(dataset)*0.8)

train_set, test_set = torch.utils.data.random_split(dataset, [len_train, len(dataset) - len_train])

#print a sample wwith label
print(train_set[0])

#dataloader
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


#model

#loss and optimizer