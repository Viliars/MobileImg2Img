import torch
import torch.nn as nn
import math

class Triple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Triple, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # down 1
        self.encoder_conv1 = Triple(in_channels, dim) # 64, 512, 512
        self.pool1 = nn.MaxPool2d(2) # 64, 256, 256
        # down 2
        self.encoder_conv2 = Triple(dim, dim*2) # 128, 256, 256
        self.pool2 = nn.MaxPool2d(2) # 128, 128, 128
        # down 3
        self.encoder_conv3 = Triple(dim*2, dim*4) # 256, 128, 128
        self.pool3 = nn.MaxPool2d(2) # 256, 64, 64
        # down 4
        self.encoder_conv4 = Triple(dim*4, dim*8) # 512, 64, 64
        self.pool4 = nn.MaxPool2d(2) # 512, 32, 32
        # center
        self.center = Triple(dim*8, dim*16) # 1024, 32, 32
        # up 1
        self.up1 = nn.ConvTranspose2d(dim*16, dim*8, 2, 2) 
        self.decoder_conv1 = Triple(dim*16, dim*8) # 512, 64, 64
        # up 2
        self.up2 = nn.ConvTranspose2d(dim*8, dim*4, 2, 2)
        self.decoder_conv2 = Triple(dim*8, dim*4) # 256, 128, 128
        # up 3
        self.up3 = nn.ConvTranspose2d(dim*4, dim*2, 2, 2)
        self.decoder_conv3 = Triple(dim*4, dim*2) # 128, 256, 256
        # up 4
        self.up4 = nn.ConvTranspose2d(dim*2, dim, 2, 2)
        self.decoder_conv4 = Triple(dim*2, dim) # 64, 512, 512
        # head
        self.head = Triple(dim, out_channels) # 3, 512, 512


    def forward(self, x):
        x1 = self.encoder_conv1(x)
        x1_half = self.pool1(x1)
        x2 = self.encoder_conv2(x1_half)
        x2_half = self.pool2(x2)
        x3 = self.encoder_conv3(x2_half)
        x3_half = self.pool3(x3)
        x4 = self.encoder_conv4(x3_half)
        x4_half = self.pool4(x4)
        x5 = self.center(x4_half)

        x6 = self.decoder_conv1(torch.cat([self.up1(x5), x4], dim=1))
        x7 = self.decoder_conv2(torch.cat([self.up2(x6), x3], dim=1))
        x8 = self.decoder_conv3(torch.cat([self.up3(x7), x2], dim=1))
        x9 = self.decoder_conv4(torch.cat([self.up4(x8), x1], dim=1))
        
        x10 = self.head(x9)

        return x10


if __name__ == "__main__":
    model = UNet(3, 3)

    x = torch.randn(1, 3, 512, 512)

    result = model(x)
    print(result.shape)