import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MeanShift

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.body(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class EResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = EResidualBlock(64, 64, group=group)
        self.b1_2 = EResidualBlock(64*2, 64*2, group=group)
        self.b1_3 = EResidualBlock(64*4, 64*4, group=group)

        self.c1 = BasicBlock(64*2, 64*2, 1, 1, 0)
        self.c2 = BasicBlock(64*4, 64*4, 1, 1, 0)
        self.c3 = BasicBlock(64*8, 64*8, 1, 1, 0)

        self.c4 = BasicBlock(512, 64, 1, 1, 0)

    def forward(self, x):
        res = c0 = o0 = x

        b1 = self.b1(o0)

        c1 = torch.cat([c0, b1], dim=1) # 128
        o1 = self.c1(c1)
        
        b2 = self.b1_2(o1) # 128, 128
        c2 = torch.cat([c1, b2], dim=1) # 128 + 128
        o2 = self.c2(c2) # 256, 256
        
        b3 = self.b1_3(o2) # 256, 256
        c3 = torch.cat([c2, b3], dim=1) # 256 + 256
        o3 = self.c3(c3)
        
        o4 = self.c4(o3)
        out = torch.add(res, o4)
        return out

class Cascade_Block(nn.Module):
    def __init__(self):
        super(Cascade_Block, self).__init__()
        self.b = Block(64, 64, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.b(x))

class MyModel(nn.Module):
    def __init__(self, num_steps):
        super(MyModel, self).__init__()

        self.num_steps = num_steps

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = MeanShift(rgb_mean, rgb_std)
        # self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        # self.block1 = BasicBlock(3, 64)
        # self.block2 = BasicBlock(64, 64)
        # self.block3 = BasicBlock(64, 3)
        self.retBlock = BasicBlock(64*2, 64, 1, 1, 0)
        self.cblock = self.make_layer(Cascade_Block, 1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))
        
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        last = x # that will go to last
        
        # x = self.sub_mean(x)

        x = self.relu(self.input(x))

        outs = []

        if torch.cuda.is_available():
            last_x = torch.zeros(x.size()).cuda()
        else:
            last_x = torch.zeros(x.size())
        
        for _ in range(self.num_steps):
            h = torch.cat((x, last_x), dim=1)
            h = self.retBlock(h)

            h = self.cblock(h)
            last_x = h

            h = self.output(h)
            h = torch.add(last, h)
            # h = self.add_mean(h)

            outs.append(h)

        return outs # return output of every timesteps
