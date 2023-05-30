import os
import yaml
import argparse
import torch
import torch.nn as nn

from model import Unet

from utils.engine import train_step, val_step
from utils.dataloaders import create_dataloaders
from utils.downloads import download_dataset
from utils.save_model import save_entire_model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


with open('data/config.yaml', 'r') as f:
    CONFIG = yaml.load(f, Loader=yaml.loader.SafeLoader)


def train(opt):
    print('[!] Trainning model')
    print('- Config: ')
    print(f'\t - Num classes: {len(CONFIG["names"])}')
    if isinstance(opt.image_size, int):
        print(f'\t - Image size: ({opt.image_size}, {opt.image_size})')
    else:
        print(f'\t - Image size: ({opt.image_size[0]}, {opt.image_size[1]})')
    print(f'\t - Epochs: {opt.epochs}')
    print(f'\t - Learning rate: {opt.learning_rate}')
    print(f'\t - Batch size: {opt.batch_size}')
    print(f'\t - Device: {device}')
    print()
    
    model = Unet(len(CONFIG['names'])).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    
    train_dir = os.path.join(CONFIG['path'], CONFIG['train'])
    train_dataloader, _ = create_dataloaders(
        dir=train_dir,
        image_size=opt.image_size,
        batch_size=opt.batch_size
    )
    
    if CONFIG['val'] is not None:
        var_dir = os.path.join(CONFIG['path'], CONFIG['val'])
        val_dataloader, _ = create_dataloaders(
            dir=var_dir,
            image_size=opt.image_size,
            batch_size=opt.batch_size
        )
    
    print('- Trainning: ')
    for epoch in range(opt.epochs):
        train_loss, train_dice, train_iou = train_step(model, train_dataloader, optimizer, criterion, device)
        
        if CONFIG['val'] is not None:
            val_loss, val_dice, val_iou = val_step(model, val_dataloader, criterion, device)
            print(f'\t- Epoch: {epoch+1}', end=' ')
            print(f'- loss: {train_loss:.4f} - dice: {train_dice:.4f} - iou: {train_iou:.4f}', end=' ')
            print(f'- val_loss: {val_loss:.4f} - val_dice: {val_dice:.4f} - val_iou: {val_iou:.4f}')
        else:
            print(f'\t- Epoch: {epoch+1}', end=' ')
            print(f'- loss: {train_loss:.4f} - dice: {train_dice:.4f} - iou: {train_iou:.4f}')
    print()
    
    save_entire_model(model, 'last')
     
     
def main(opt):
    try:
        if CONFIG['download']:
            download_dataset()
    except:
        pass

    train(opt)
       
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, help='num epochs')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--image_size', type=int, nargs='+', default=256, help='image size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)