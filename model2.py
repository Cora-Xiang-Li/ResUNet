"""
U-Net part from: https://github.com/Jinglever/pytorch-unet.git
"""
"""
https:*arxiv.org/abs/1611.05431
official code:
https:*github.com/facebookresearch/ResNeXt
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

from utils import Model_Logger

logger = Model_Logger('model')
logger.enable_exception_hook()

class UNetEncoder(nn.Module):
    def __init__(self, blocks):
        super(UNetEncoder, self).__init__()
        assert len(blocks) > 0
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        skips = []
        for i in range(1, len(self.blocks) - 1):
            x = self.blocks[i](x)
            skips.append(x)
        res = [self.blocks[i+1](x)]
        res += skips
        return res

class UNetDecoder(nn.Module):
    def __init__(self, blocks):
        super(UNetDecoder, self).__init__()
        assert len(blocks) > 1
        self.blocks = nn.ModuleList(blocks)
    
    def _center_crop(self, skip, x):
        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self, x, skips, reverse_skips=True):
        assert len(skips) == len(self.blocks) - 1
        if reverse_skips:
            skips = skips[::-1]
        x = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            skip, x = self._center_crop(skips[i-1], x)
            x = torch.cat([skip, x], dim=1)
            x = self.blocks[i](x)
        return x

class UNetFactory(nn.Module):
    def __init__(self, encoder_blocks, decoder_blocks, bridge=None):
        super(UNetFactory, self).__init__()
        self.encoder = UNetEncoder(encoder_blocks)
        self.bridge = bridge
        self.decoder = UNetDecoder(decoder_blocks)

    def forward(self, x):
        res = self.encoder(x)
        out, skips = res[0], res[1:]
        if self.bridge is not None:
            out = self.bridge(out)
        out = self.decoder(out, skips)
        return out
    
def unet_convs(in_channels, out_channels, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def unet(in_channels, out_channels):
    encoder_blocks = [
        unet_convs(in_channels, 64),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            unet_convs(64, 128)
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            unet_convs(128, 256)
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            unet_convs(256, 512)
        ),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    ]
    bridge = nn.Sequential(
        unet_convs(512, 1024)
    )
    decoder_blocks = [
        nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        nn.Sequential(
            unet_convs(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        ),
        nn.Sequential(
            unet_convs(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        ),
        nn.Sequential(
            unet_convs(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        ),
        nn.Sequential(
            unet_convs(128, 64),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
    ]
    return UNetFactory(encoder_blocks, decoder_blocks, bridge)

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
        # out = self.pool0(out)
        out1 = self.layer1(out)
        out1 = self.shape_out(out1)

        out2 = self.layer2(out1)
        out2 = self.shape_out(out2)

        out3 = self.layer3(out2)
        out3 = self.shape_out(out3)

        out4 = self.layer4(out3)
        out4 = self.shape_out(out4)
        
        return out1, out2, out3, out4

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

def resnext50_16x4d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=16, bottleneck_width=4)

def resnext50_2x64d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=8)

def resnext50_32x4d():
    return ResNeXt(num_blocks=[3, 4, 10, 3], cardinality=32, bottleneck_width=8)


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
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(256*2, 128*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(64*k, 32*k, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32*k, momentum=momentum),
            nn.GELU()
        )
        self.up1_ = nn.Conv2d(64*k, 32*k, 1)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(512*2, 256*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(128*k, 64*k, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64*k),
            nn.GELU()
        )
        self.up2_ = nn.Conv2d(128*k, 64*k, 1)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(1024*2, 512*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(256*k, 128*k, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128*k),
            nn.GELU()
        )
        self.up3_ = nn.Conv2d(256*k, 128*k, 1)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(2048*2, 1024*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(512*k, 256*k, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256*k),
            nn.GELU()
        )
        self.up4_ = nn.Conv2d(512*k, 256*k, 1)

        self.cls_seg = nn.Conv2d(64, 1, 1)

        # self.out = OutConv(64, 1)
        
        if batch_normalize:
            self.bn = nn.BatchNorm2d(128, momentum=momentum)
        else:
            self.bn = None

    def forward(self, x):
        xx1, xx2, xx3, xx4 = self.backbone(x)
        
        stage1, stage2, stage3, stage4 = xx1, xx2, xx3, xx4
        up4 = self.up4(stage4)
        up4 = torch.cat([up4, stage3], dim=1)
        up4 = self.up4_(up4)

        up3 = self.up3(up4)
        up3 = torch.cat([up3, stage2], dim=1)
        up3 = self.up3_(up3)

        up2 = self.up2(up3)
        up2 = torch.cat([up2, stage1], dim=1)
        up2 = self.up2_(up2)

        out2 = self.up1(up2)

        out2 = self.cls_seg(out2)

        out1 = self.unet(x)
        out = out1*0.4+out2*0.6

        return out
    
def unet_resnet(resnet_type='resnext50_32x4d', in_channels=128, out_channels=1, pretrained=True):
    if resnet_type == 'resnet18':
        resnet = torchvision.models.resnet.resnet18(pretrained)
        encoder_out_channels = [in_channels, 64, 64, 128, 256, 512]
    elif resnet_type == 'resnet34':
        resnet = torchvision.models.resnet.resnet34(pretrained)
        encoder_out_channels = [in_channels, 64, 64, 128, 256, 512]
    elif resnet_type == 'resnet50':
        resnet = torchvision.models.resnet.resnet50(pretrained)
        encoder_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    elif resnet_type == 'resnet101':
        resnet = torchvision.models.resnet.resnet101(pretrained)
        encoder_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    elif resnet_type == 'resnet152':
        resnet = torchvision.models.resnet.resnet152(pretrained)
        encoder_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    elif resnet_type == 'resnext50_32x4d':
        resnet = torchvision.models.resnet.resnext50_32x4d(pretrained)
        encoder_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    else:
        raise ValueError("unexpected resnet_type")

    encoder_blocks = [
        nn.Sequential(),
        nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resnet.layer1
        ),
        resnet.layer2,
        resnet.layer3,
        resnet.layer4
    ]
    bridge = None
    decoder_blocks = []
    in_ch = encoder_out_channels[-1]
    out_ch = in_ch // 2
    decoder_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
    for i in range(1, len(encoder_blocks)-1):
        in_ch = encoder_out_channels[-i-1] + out_ch
        decoder_blocks.append(nn.Sequential(
            unet_convs(in_ch, out_ch, padding=1),
            nn.ConvTranspose2d(out_ch, out_ch//2, kernel_size=2, stride=2),
        ))
        out_ch = out_ch // 2
    in_ch = encoder_out_channels[0] + out_ch
    decoder_blocks.append(nn.Sequential(
        unet_convs(in_ch, out_ch, padding=1),
        nn.Conv2d(out_ch, out_channels, kernel_size=1)
    ))

    return UNetFactory(encoder_blocks, decoder_blocks, bridge)
