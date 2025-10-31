import torch
import torch.nn as nn

class ThreeDUNet(nn.Module):
    """
    """
    def __init__(self, in_channels=1, out_channels=6):
        super().__init__()

        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)

        self.enc4 = self._conv_block(128, 256)

        # Decoder (upsampling)
        self.dec4 = self._conv_block(256 + 128, 128)
        self.dec3 = self._conv_block(128 + 64, 64)
        self.dec2 = self._conv_block(64 + 32, 32)
        self.dec1 = nn.Conv3d(32, out_channels, 1)

        self.pool = nn.MaxPool3d(2, stride=2)
        self.up1 = nn.ConvTranspose3d(256, 256, 2, 2)
        self.up2 = nn.ConvTranspose3d(128, 128, 2, 2)
        self.up3 = nn.ConvTranspose3d(64, 64, 2, 2)

    def _conv_block(self, in_ch, out_ch):
        """Conv block with batch normalization and LeakyReLU: Conv -> BN -> LeakyReLU -> Conv -> BN -> LeakyReLU"""
        return nn.Sequential(
            nn.Conv3d(in_ch, int(out_ch//2), 3, padding=1),
            nn.BatchNorm3d(int(out_ch//2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(int(out_ch//2), out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x) 
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))


        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up1(e4), e3], 1))
        d3 = self.dec3(torch.cat([self.up2(d4), e2], 1))
        d2 = self.dec2(torch.cat([self.up3(d3), e1], 1))
        out = self.dec1(d2)

        return out