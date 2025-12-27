# src/model.py
import torch
import torch.nn as nn

class FastUNet(nn.Module):
    """
    Extremely lightweight U-Net for fast training
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder (upsampling)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = self.conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self.conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = self.conv_block(64, 32)

        # Output
        self.out = nn.Conv2d(32, out_channels, 1)

        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out(d1))


class FocalDice(nn.Module):
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, gt):
        pred = pred.view(-1)
        gt = gt.view(-1)
        inter = (pred * gt).sum()
        dice = (2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
        return (1 - dice) ** self.gamma


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = FocalDice()

    def forward(self, pred, gt):
        has_fg = (gt.sum(dim=(1, 2, 3)) > 0).float()
        bce = self.bce(pred, gt)
        dice = self.dice(pred, gt)
        loss = has_fg * (0.5 * bce + 0.5 * dice) + (1 - has_fg) * bce * 0.3
        return loss.mean()