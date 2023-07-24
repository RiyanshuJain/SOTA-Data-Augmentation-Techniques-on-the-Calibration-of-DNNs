import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, lr_scheduler
from torchvision.datasets import CIFAR10
from torchvision import transforms

from datetime import datetime
import random
from tqdm import tqdm
from PIL import Image
from PIL import ImageOps
import warnings
warnings.filterwarnings('ignore')

# from resnet import resnet32x4
# from wideresnet import wrn_40_2
# from . import model_dict
from utils import compute_calibration_metrics

# print(model_dict)
torch.cuda.is_available()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

__all__ = ['wrn']

class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.avgpool = nn.AvgPool2d(8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def distill_seq(self):
        feat_m = nn.ModuleList([])  
        #feat_m.append(self.conv1)       
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(nn.Sequential(
            self.block3, self.bn1, self.relu))
        feat_m.append(nn.Sequential(
            self.avgpool,
            #Rearrange('b h w -> (b h w) 1'),
            Flatten(),
            self.fc))
        return feat_m

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        # f3 = out
        out = self.relu(self.bn1(out))
        f3 = out
        out = self.avgpool(out)
        out = out.view(-1, self.nChannels)
        f4 = out
        out = self.fc(out)
        if is_feat:
            if preact:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)
            return [f0, f1, f2, f3, f4], out
        else:
            return out


def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model


def wrn_40_2(**kwargs):
    model = WideResNet(depth=40, widen_factor=2, **kwargs)
    return model

mean = (0.4915, 0.4823, 0.4468)
std_dev = (0.2470, 0.2435, 0.2616)

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std_dev)
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std_dev)
    ])

batch_size = 64

distortions = ['brightness', 'contrast', 'defocus_blur', 
                'elastic_transform', 'fog', 'frost', 'gaussian_blur', 
                'gaussian_noise', 'glass_blur', 'impulse_noise', 
                'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 
                'shot_noise', 'snow', 'spatter','speckle_noise', 'zoom_blur']

def validate_curropted(distortions, test_d, test_loader, net, s):
    print("Validating {} wrn_40-2 model...".format(s))
    ece, bin_acc, bin_conf, acc, bin_count = compute_calibration_metrics(num_bins=100, net=net, loader=test_loader)
    print(ece, acc)

    print("\n------------------------------\n")
    avg_acc = 0.0
    avg_ece = 0.0

    for dist in distortions:
        print(f"Validating {dist} curroption...")
        test_d.data = np.load(f'./CIFAR-10-C/{dist}.npy')
        test_d.targets =  torch.LongTensor(np.load('./CIFAR-10-C/labels.npy'))
        test_loader1 = torch.utils.data.DataLoader(test_d,batch_size=batch_size, shuffle=True, num_workers=8)

        ece, bin_acc, bin_conf, acc, bin_count = compute_calibration_metrics(num_bins=100, net=net, loader=test_loader1)
        print(ece, acc)

        avg_acc += acc
        avg_ece += ece

    avg_acc /= len(distortions)
    avg_ece /= len(distortions)
    print("Average accuracy: ", avg_acc)
    print("Average ece: ", avg_ece)
    print("------------------------")

for i in ['without', 'mixup', 'cutout', 'cutmix', 'all_in_one', 'sequentially']:
    test_d = CIFAR10(root='../cifar10/data/', train=False, transform=transform_test, download=True)
    test_loader = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=True, num_workers=8)
    model = wrn_40_2(num_classes = 10)
    x = torch.load('../cifar10/weights/wrn40_2/{}/best_ckpt'.format(i))
    model.load_state_dict(x['state_dict'])
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    validate_curropted(distortions, test_d, test_loader, model, i)