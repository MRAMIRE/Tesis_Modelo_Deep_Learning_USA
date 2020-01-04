import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, int(64))
        self.down1 = Down(int(64), int(128))
        self.down2 = Down(int(128), int(256))
        self.down3 = Down(int(256), int(512))
        self.down4 = Down(int(512), int(512))
        self.up1 = Up(int(1024), int(256), bilinear)
        self.up2 = Up(int(512), int(128), bilinear)
        self.up3 = Up(int(256), int(64), bilinear)
        self.up4 = Up(int(128), int(64), bilinear)
        self.outc = OutConv(int(64), n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits