import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.block(x)
    
    
class Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block_down_1 = UnetBlock(3, 64)
        self.block_down_2 = UnetBlock(64, 128)
        self.block_down_3 = UnetBlock(128, 256)
        self.block_down_4 = UnetBlock(256, 512)
        
        self.block_neck = UnetBlock(512, 1024)
        
        self.block_up_1 = UnetBlock(1024 + 512, 512)
        self.block_up_2 = UnetBlock(512 + 256, 256)
        self.block_up_3 = UnetBlock(256 + 128, 128)
        self.block_up_4 = UnetBlock(128 + 64, 64)
        
        self.conv_cls = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x_down_1 = self.block_down_1(x)
        x = F.max_pool2d(x_down_1, kernel_size=2)
        
        x_down_2 = self.block_down_2(x)
        x = F.max_pool2d(x_down_2, kernel_size=2)
        
        x_down_3 = self.block_down_3(x)
        x = F.max_pool2d(x_down_3, kernel_size=2)
        
        x_down_4 = self.block_down_4(x)
        x = F.max_pool2d(x_down_4, kernel_size=2)
        
        x = self.block_neck(x)
        
        x_up_1 = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = torch.cat([x_down_4, x_up_1], dim=1)
        x = self.block_up_1(x)
        
        x_up_2 = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = torch.cat([x_down_3, x_up_2], dim=1)
        x = self.block_up_2(x)
        
        x_up_3 = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = torch.cat([x_down_2, x_up_3], dim=1)
        x = self.block_up_3(x)
        
        x_up_4 = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = torch.cat([x_down_1, x_up_4], dim=1)
        x = self.block_up_4(x)
        
        return self.conv_cls(x)