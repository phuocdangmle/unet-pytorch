import yaml
import torch
import torchmetrics
from utils.metrics import AverageMeter
device = "cuda:0" if torch.cuda.is_available() else "cpu"


with open("data/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    
num_classes = len(config["names"])

dice = torchmetrics.Dice(num_classes=num_classes, average="macro").to(device)
iou = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass", average="macro").to(device)


def train_step(model, train_dataloader, optimizer, criterion, device):
    train_loss = AverageMeter()
    train_dice = AverageMeter()
    train_iou = AverageMeter()
    
    model.train()
    
    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)
        
        logits = model(X)
        
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        n = X.shape[0]
        train_loss.update(loss.item(), n)
        masks = logits.argmax(dim=1).squeeze(dim=1) # B, C, H, W -> B, 1, H, W -> B, H, W
        train_dice.update(dice(masks, y).item(), n)
        train_iou.update(iou(masks, y).item(), n)
        
    return train_loss.avg, train_dice.avg, train_iou.avg


def val_step(model, val_dataloader, criterion, device):    
    val_loss = AverageMeter()
    val_dice = AverageMeter()
    val_iou = AverageMeter()
    
    model.eval()
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(val_dataloader):
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            
            loss = criterion(logits, y)
            
            
            n = X.shape[0]
            val_loss.update(loss.item(), n)
            masks = logits.argmax(dim=1).squeeze(dim=1) # B, C, H, W -> B, 1, H, W -> B, H, W
            val_dice.update(dice(masks, y).item(), n)
            val_iou.update(iou(masks, y).item(), n)
        
    return val_loss.avg, val_dice.avg, val_iou.avg