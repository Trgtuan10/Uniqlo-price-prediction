import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import argument_parser, batch_trainer, valid_trainer, batch_trainner_regression, valid_trainer_regression
from datasets.UniqloDataset import UniqloDataset, get_transform
import torchvision.transforms as T
from model.resnet import resnet50, resnet101, resnext50_32x4d,resnet152,resnet18,resnet34
from model.price_model import *
from torch.utils.data import ConcatDataset
import wandb


def main(args):
    print(f'use GPU{args.device} for training')

    train_tsfm, valid_tsfm = get_transform(args)

    num_augmented_copies = 5

    augmented_datasets = []
    for _ in range(num_augmented_copies):
        augmented_datasets.append(UniqloDataset(csv_file='datasets/csv_for_price/image_data.csv', 
                                                root_dir='datasets/images', 
                                                label_column='Price',
                                                transform=train_tsfm))


    train_set = ConcatDataset(augmented_datasets)

    valid_set = UniqloDataset(csv_file='datasets/csv_for_price/image_valid.csv', 
                              root_dir='datasets/images',
                              label_column='Price',
                              transform=valid_tsfm)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.workers,
    )
    
    # print(f'train set: {len(train_set)} samples')

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.workers,
    )
    
    # print(f'valid set: {len(valid_set)} samples')
    #model
    backbone = resnet18()
    model = Uniqlo(backbone)

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.use_pretrain==True:
        model_path = args.pretrain_model  # luu dia chi cua model
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
    
    
    # save checkpoint
    exp_dir = args.checkpoint
    #name with date
    time_str = time.strftime("%Y%m%d-%H%M%S")
    last_checkpoint_filename = f'price_model_lr_{args.lr_ft}_{args.lr_new}_{time_str}.pt'
    last_checkpoint_path = os.path.join(exp_dir, last_checkpoint_filename)

    criterion = nn.MSELoss()
    param_groups = [{'params': model.finetune_params(), 'lr': args.lr_ft},
                    {'params': model.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    # Initialize Weights & Biases
    if args.log:
        wandb.init(project="Uniqlo price prediction", name = f'Uniqlo_price_lr_{args.lr_ft}_{args.lr_new}_batch_size_{args.batchsize}')

    #training
    for i in range(args.train_epoch):
        print(f'start epoch :  {i+1}')
        train = batch_trainner_regression(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid = valid_trainer_regression(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        lr_scheduler.step(metrics=valid[0])
        
        #log
        if args.log:
            wandb.log({
            "Train Loss": train[0], 
            "Train Accuracy": train[1],
            "Validation Loss": valid[0], 
            "Validation Accuracy": valid[1],
            })

        print("-----------------------------------------------------------------------------------------------------------")

        # save checkpoint
        if (i + 1) % 20 == 0:
            save_checkpoint(model, optimizer, i + 1, last_checkpoint_path)
            print("Replacing last checkpoint with the new one.")


def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    if not os.path.exists(filepath):
        torch.save(checkpoint, filepath)
    else:
        os.remove(filepath)
        torch.save(checkpoint, filepath)  

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)