import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BasicBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


### RSU-7 ###
class RSU7(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU7, self).__init__()

        self.first_conv = BasicBlock(in_channels, out_channels)

        self.block1 = BasicBlock(out_channels, mid_channels)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.block2 = BasicBlock(mid_channels, mid_channels)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.block3 = BasicBlock(mid_channels, mid_channels)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.block4 = BasicBlock(mid_channels, mid_channels)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.block5 = BasicBlock(mid_channels, mid_channels)
        self.pool5 = nn.MaxPool2d(2, stride=2)

        self.block6 = BasicBlock(mid_channels, mid_channels)

        self.block7 = BasicBlock(mid_channels, mid_channels)

        self.block6d = BasicBlock(mid_channels*2, mid_channels)

        self.up6 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block5d = BasicBlock(mid_channels*2, mid_channels)

        self.up5 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block4d = BasicBlock(mid_channels*2, mid_channels)

        self.up4 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block3d = BasicBlock(mid_channels*2, mid_channels)

        self.up3 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block2d = BasicBlock(mid_channels*2, mid_channels)

        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block1d = BasicBlock(mid_channels*2, out_channels)

    def forward(self, x):
        hx = x
        hx_in = self.first_conv(hx)

        hx1 = self.block1(hx_in)
        hx = self.pool1(hx1)

        hx2 = self.block2(hx)
        hx = self.pool2(hx2)

        hx3 = self.block3(hx)
        hx = self.pool3(hx3)

        hx4 = self.block5(hx)
        hx = self.pool4(hx4)

        hx5 = self.block6(hx)
        hx = self.pool5(hx5)

        hx6 = self.block6(hx)

        hx7 = self.block7(hx6)

        hx6d =  self.block6d(torch.cat((hx7, hx6), dim=1))

        hx6dup = self.up6(hx6d)
        hx5d =  self.block5d(torch.cat((hx6dup, hx5), dim=1))

        hx5dup = self.up5(hx5d)
        hx4d = self.block4d(torch.cat((hx5dup, hx4), dim=1))

        hx4dup = self.up4(hx4d)
        hx3d = self.block3d(torch.cat((hx4dup, hx3), dim=1))

        hx3dup = self.up3(hx3d)
        hx2d = self.block2d(torch.cat((hx3dup, hx2), dim=1))

        hx2dup = self.up2(hx2d)
        hx1d = self.block1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hx_in


### RSU-6 ###
class RSU6(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU6, self).__init__()

        self.first_conv = BasicBlock(in_channels, out_channels)

        self.block1 = BasicBlock(out_channels, mid_channels)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.block2 = BasicBlock(mid_channels, mid_channels)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.block3 = BasicBlock(mid_channels, mid_channels)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.block4 = BasicBlock(mid_channels, mid_channels)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.block5 = BasicBlock(mid_channels, mid_channels)

        self.block6 = BasicBlock(mid_channels, mid_channels)

        self.block5d = BasicBlock(mid_channels*2, mid_channels)

        self.up5 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block4d = BasicBlock(mid_channels*2, mid_channels)

        self.up4 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block3d = BasicBlock(mid_channels*2, mid_channels)

        self.up3 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block2d = BasicBlock(mid_channels*2, mid_channels)

        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block1d = BasicBlock(mid_channels*2, out_channels)

    def forward(self, x):
        hx = x
        hx_in = self.first_conv(hx)

        hx1 = self.block1(hx_in)
        hx = self.pool1(hx1)

        hx2 = self.block2(hx)
        hx = self.pool2(hx2)

        hx3 = self.block3(hx)
        hx = self.pool3(hx3)

        hx4 = self.block5(hx)
        hx = self.pool4(hx4)

        hx5 = self.block6(hx)

        hx6 = self.block6(hx5)

        hx5d =  self.block5d(torch.cat((hx6, hx5), dim=1))

        hx5dup = self.up5(hx5d)
        hx4d =  self.block4d(torch.cat((hx5dup, hx4), dim=1))

        hx4dup = self.up4(hx4d)
        hx3d = self.block3d(torch.cat((hx4dup, hx3), dim=1))

        hx3dup = self.up3(hx3d)
        hx2d = self.block2d(torch.cat((hx3dup, hx2), dim=1))

        hx2dup = self.up2(hx2d)
        hx1d = self.block1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hx_in


### RSU-5 ###
class RSU5(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU5, self).__init__()

        self.first_conv = BasicBlock(in_channels, out_channels)

        self.block1 = BasicBlock(out_channels, mid_channels)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.block2 = BasicBlock(mid_channels, mid_channels)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.block3 = BasicBlock(mid_channels, mid_channels)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.block4 = BasicBlock(mid_channels, mid_channels)
 
        self.block5 = BasicBlock(mid_channels, mid_channels)

        self.block4d = BasicBlock(mid_channels*2, mid_channels)

        self.up4 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block3d = BasicBlock(mid_channels*2, mid_channels)

        self.up3 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block2d = BasicBlock(mid_channels*2, mid_channels)

        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block1d = BasicBlock(mid_channels*2, out_channels)

    def forward(self, x):
        hx = x
        hx_in = self.first_conv(hx)

        hx1 = self.block1(hx_in)
        hx = self.pool1(hx1)

        hx2 = self.block2(hx)
        hx = self.pool2(hx2)

        hx3 = self.block3(hx)
        hx = self.pool3(hx3)

        hx4 = self.block4(hx)

        hx5 = self.block5(hx4)

        hx4d =  self.block4d(torch.cat((hx5, hx4), dim=1))

        hx4dup = self.up4(hx4d)
        hx3d = self.block3d(torch.cat((hx4dup, hx3), dim=1))

        hx3dup = self.up3(hx3d)
        hx2d = self.block2d(torch.cat((hx3dup, hx2), dim=1))

        hx2dup = self.up2(hx2d)
        hx1d = self.block1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hx_in


### RSU-4 ###
class RSU4(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU4, self).__init__()

        self.first_conv = BasicBlock(in_channels, out_channels)

        self.block1 = BasicBlock(out_channels, mid_channels)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.block2 = BasicBlock(mid_channels, mid_channels)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.block3 = BasicBlock(mid_channels, mid_channels)

        self.block4 = BasicBlock(mid_channels, mid_channels)

        self.block3d = BasicBlock(mid_channels*2, mid_channels)

        self.up3 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block2d = BasicBlock(mid_channels*2, mid_channels)

        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, 2)
        self.block1d = BasicBlock(mid_channels*2, out_channels)

    def forward(self, x):
        hx = x
        hx_in = self.first_conv(hx)

        hx1 = self.block1(hx_in)
        hx = self.pool1(hx1)

        hx2 = self.block2(hx)
        hx = self.pool2(hx2)

        hx3 = self.block3(hx)

        hx4 = self.block4(hx3)

        hx3d = self.block3d(torch.cat((hx4, hx3), dim=1))

        hx3dup = self.up3(hx3d)
        hx2d = self.block2d(torch.cat((hx3dup, hx2), dim=1))

        hx2dup = self.up2(hx2d)
        hx1d = self.block1d(torch.cat((hx2dup, hx1), dim=1))

        return hx1d + hx_in


class LinearMapper(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(LinearMapper, self).__init__()
        self.net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16384, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 16384),
        nn.Unflatten(1, (64, 16, 16))
    )

    def forward(self, x):
        out = self.net(x)
        return out

##### U^2-Net ####
class U2Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(U2Net,self).__init__()

        self.first_conv = BasicBlock(in_channels, 64)

        self.stage1 = RSU7(64, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2,stride=2)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2)

        self.stage5 = RSU4(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2)

        self.stage6 = RSU4(64, 16, 64)

        self.mapper = LinearMapper()

        # decoder
        self.up6 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.stage5d = RSU4(128, 16, 64)
        self.up5 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.stage4d = RSU4(128, 16, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.stage3d = RSU5(128, 16, 64)
        self.up3 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.stage2d = RSU6(128, 16, 64)
        self.up2 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.stage1d = RSU7(128, 16, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 3, 1, 1)

    def forward(self,x):

        hx = x
        hx_in = self.first_conv(hx)

        #stage 1
        hx1 = self.stage1(hx_in)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)

        mid = self.mapper(hx6)
        
        hx6up = self.up6(mid)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), dim=1))
        hx5dup = self.up5(hx5d)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), dim=1))
        hx4dup = self.up4(hx4d)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), dim=1))
        hx3dup = self.up3(hx3d)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), dim=1))
        hx2dup = self.up2(hx2d)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), dim=1))

        result = self.conv_last(hx1d + hx_in)

        return result


if __name__ == "__main__":
    model = RSU7(3, 12, 3)
    
    x = torch.rand((1, 3, 512, 512))

    result = model(x)
    assert result.shape == torch.Size([1, 3, 512, 512])

    model = RSU6(3, 12, 3)

    x = torch.rand((1, 3, 512, 512))

    result = model(x)
    assert result.shape == torch.Size([1, 3, 512, 512])

    model = RSU5(3, 12, 3)

    x = torch.rand((1, 3, 512, 512))

    result = model(x)
    assert result.shape == torch.Size([1, 3, 512, 512])

    model = RSU4(3, 12, 3)

    x = torch.rand((1, 3, 512, 512))

    result = model(x)
    assert result.shape == torch.Size([1, 3, 512, 512])

    model = U2Net(3, 3)
    
    x = torch.rand((1, 3, 512, 512))

    result = model(x)
    assert result.shape == torch.Size([1, 3, 512, 512])

    print("-------------------------")

    model = U2Net(3, 3)
    
    x = torch.rand((1, 3, 512, 512))
    model.eval()
    model = model.cuda()
    x = x.cuda()

    for i in range(20):
        start = time.time()
        result = model(x)
        end = time.time()
        print(end - start)





