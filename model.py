"""
https:*arxiv.org/abs/1611.05431
official code:
https:*github.com/facebookresearch/ResNeXt

After changing decoder process, this model is perfectly running with resNext and U-Net architecture
Modified date: 2024/08/12
Modified Author: Cora Li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# from torch.autograd import Variable
import numpy as np

from utils import Model_Logger

logger = Model_Logger('model')
logger.enable_exception_hook()

class BasicBlock_C(nn.Module):
    """
    increasing cardinality is a more effective way of 
    gaining accuracy than going deeper or wider
    """

    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, stride=1, expansion=2):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', nn.Conv2d(in_planes, inner_width, 1, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(inner_width)),
                ('act0', nn.ReLU()),
                ('conv3_0', nn.Conv2d(inner_width, inner_width, 3, stride=stride, padding=1, groups=cardinality, bias=False)),
                ('bn2', nn.BatchNorm2d(inner_width)),
                ('act1', nn.ReLU()),
                ('conv1_1', nn.Conv2d(inner_width, inner_width * self.expansion, 1, stride=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inner_width * self.expansion))
            ]
        ))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, inner_width * self.expansion, 1, stride=stride, bias=False)
            )
        self.bn0 = nn.BatchNorm2d(self.expansion * inner_width)

    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)
        out = F.relu(self.bn0(out))
        return out

class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2, num_classes=2):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 128
        self.expansion = expansion
        
        # Given groups=1, weight of size [64, 3, 3, 3], expected input[8, 1, 128, 128] to have 3 channels, but got 1 channels instead
        self.conv0 = nn.Conv2d(1, self.in_planes, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=self._make_layer(num_blocks[0],1)
        self.layer2=self._make_layer(num_blocks[1],2)
        self.layer3=self._make_layer(num_blocks[2],2)
        self.layer4=self._make_layer(num_blocks[3],2)
        self.linear = nn.Linear(self.cardinality * self.bottleneck_width, num_classes)

    def shape_out(self, out_n):
        out_n = F.avg_pool2d(out_n, 2)
        # out_n = out_n.view(out_n.size(0), -1)
        # out_n = self.linear(out_n)
        return out_n
    
    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        out1 = self.shape_out(out1)
        out2 = self.shape_out(out2)
        out3 = self.shape_out(out3)
        out4 = self.shape_out(out4)
        
        # out = self.pool0(out)

        return out, out1, out2, out3, out4

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion))
            self.in_planes = self.expansion * self.bottleneck_width * self.cardinality
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

def resnext29_8x64d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=8, bottleneck_width=64)

def resnext29_16x64d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=16, bottleneck_width=64)

def resnext29_32x4d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4)

def resnext34_32x4d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=16, bottleneck_width=4)

class ResUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=True, momentum=0.9, batch_normalize=True):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.backbone = resnext34_32x4d()
        self.bilinear = bilinear
        self.pool = nn.MaxPool2d((2, 2))
        # self.conv = DoubleConv(n_channels, 64)
        k = 2
        self.upConv1 = nn.ConvTranspose2d(64*k, 32*2*k, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64*2*k, 32*k, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32*k, momentum=momentum),
            nn.GELU()
        )

        self.upConv2 = nn.ConvTranspose2d(128*k, 64*k, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128*k, 64*k, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64*k),
            nn.GELU()
        )

        self.upConv3 = nn.ConvTranspose2d(256*k, 128*k, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256*k, 128*k, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128*k),
            nn.GELU()
        )

        self.upConv4 = nn.ConvTranspose2d(512*k, 256*k, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512*k, 256*k, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256*k),
            nn.GELU()
        )

        self.cls_seg = nn.Conv2d(64, 1, 1)
        
        if batch_normalize:
            self.bn = nn.BatchNorm2d(128, momentum=momentum)
        else:
            self.bn = None

    def extract_features(self, x):
        x, _, _, _, _ = self.backbone(x)
        return x
    
    def forward(self, x):
        x, stage1, stage2, stage3, stage4 = self.backbone(x)

        up4 = self.upConv4(stage4)
        up4 = torch.cat([up4, stage3], dim=1)
        up4 = self.decoder4(up4)

        up3 = self.upConv3(up4)
        up3 = torch.cat([up3, stage2], dim=1)
        up3 = self.decoder3(up3)

        up2 = self.upConv2(up3)
        up2 = torch.cat([up2, stage1], dim=1)
        up2 = self.decoder2(up2)

        up1 = self.upConv1(up2)
        up1 = torch.cat([up1, x], dim=1)
        up1 = self.decoder1(up1)

        out = self.cls_seg(up1)

        # m = nn.Sigmoid()
        # out = m(out)
        # out[out < 0.65] = 0
        # out_indices = torch.nonzero(out > 0.65, as_tuple=False)
        # out_cord = out_indices[:, [3, 2]]

        return out, x