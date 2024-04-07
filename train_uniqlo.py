import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import argument_parser, batch_trainer, valid_trainer
from datasets.UniqloDataset import UniqloDataset, get_transform
import torchvision.transforms as T
from resnet import resnet50, resnet101, resnext50_32x4d,resnet152,resnet18,resnet34
from Uniqlo import *


def main(args):
    print(f'use GPU{args.device} for training')

    train_tsfm, valid_tsfm = get_transform(args)

    train_set = UniqloDataset(csv_file='datasets/image_data.csv', 
                        root_dir='datasets/images', 
                        transform=train_tsfm)
    #Them csv valid vao day
    valid_set = UniqloDataset(csv_file='datasets/image_valid.csv', 
                        root_dir='datasets/images', 
                        transform=valid_tsfm)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.workers,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.workers,
    )

    #model
    backbone = resnet18()
    model = Uniqlo(backbone)

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.use_pretrain==True:
        model_path = args.pretrain_model  # luu dia chi cua model
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dicts'])
    
    # save checkpoint
    exp_dir = args.checkpoint
    last_checkpoint_filename = 'last.pt'
    last_checkpoint_path = os.path.join(exp_dir, last_checkpoint_filename)

    criterion = nn.MSELoss()
    param_groups = [{'params': model.finetune_params(), 'lr': args.lr_ft},
                    {'params': model.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    #training
    for i in range(args.train_epoch):
        train_loss = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        valid_loss = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        lr_scheduler.step(metrics=valid_loss)

        # save checkpoint
        if (i + 1) % 10 == 0:
            save_checkpoint(model, optimizer, i + 1, last_checkpoint_path)
            print("Replacing last checkpoint with the new one.")


def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if not os.path.exists(filepath):
        print("last.pt not found. Creating a new checkpoint.")
        torch.save(checkpoint, filepath)
    else:
        torch.save(checkpoint, filepath + '.tmp') 
        os.replace(filepath + '.tmp', filepath)  



if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)