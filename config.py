import argparse
import time
import torch
import datetime
import numpy as np
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--train_epoch", type=int, default=500)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256) 
    parser.add_argument("--workers", type=int, default=4) 
    parser.add_argument("--lr_ft", type=float, default=0.01, help='learning rate of feature extractor')
    parser.add_argument("--lr_new", type=float, default=0.1, help='learning rate of classifier_base')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', default="0", type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')
    parser.add_argument('--use_pretrain', default=False, type=bool)
    parser.add_argument('--pretrain_model', default="path_to_model.pt", type=str)
    parser.add_argument('--checkpoint', default="./exp", type=str)
    parser.add_argument('--log', type=bool, default=False)
    return parser


class AverageMeter(object):
    """ 
    Computes and stores the average and current value

    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-20)

def batch_trainer(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_num = len(train_loader)

    lr = optimizer.param_groups[0]['lr']
    
    for step, (imgs, gt_label) in enumerate(train_loader):
        batch_time = time.time()
        # Move data to device
        imgs, gt_label = imgs.to(device), gt_label.to(device)
        
        # Forward pass
        train_predict = model(imgs)
        
        # Calculate accuracy
        _, predicted = torch.max(train_predict, 1)
        correct = predicted.eq(gt_label).sum().item()
        acc_meter.update(correct / imgs.size(0))
        
        print(f'predicted: {predicted}')
        print(f'gt_label: {gt_label}')
        print(f'correct: {correct}')
        print(f'acc_meter: {acc_meter.avg}')
        
        # Calculate loss
        train_loss = criterion(train_predict, gt_label)
        
        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        loss_meter.update(train_loss.item(), imgs.size(0))

        log_interval = 20
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {time.time() - batch_time:.2f}s ',
                  f'train_loss:{loss_meter.val:.4f}')

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}, Accuracy: {acc_meter.avg:.2%}')

    return loss_meter.avg, acc_meter.avg


# @torch.no_grad()
def valid_trainer(model, valid_loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        total_samples = 0
        correct_predictions = 0
        for step, (imgs, gt_label) in enumerate(tqdm(valid_loader)):
            imgs = imgs.to(device)
            gt_label = gt_label.to(device)
            valid_predict = model(imgs)
            
            # Calculate number of correct predictions
            _, predicted = torch.max(valid_predict, 1)
            correct_predictions += predicted.eq(gt_label).sum().item()
            
            # Count total samples
            total_samples += imgs.size(0)
            
            valid_loss = criterion(valid_predict, gt_label)
            loss_meter.update(valid_loss.item(), imgs.size(0))
        
        accuracy_percentage = (correct_predictions / total_samples) * 100
        print(f'Accuracy: {accuracy_percentage:.2f}%')
    valid_loss = loss_meter.avg
    return valid_loss, accuracy_percentage

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)