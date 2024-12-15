#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
# 所有神经网络的基类torch.nn
from addition_experiment.conv2d import ConvBlock
# conv+bn+relu


class MLP(nn.Module):
    def __init__(self, dim_in=1, dim_hidden=64, dim_out=10):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class CNNMnist(nn.Module):
    def __init__(self, args=None):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNN__mnist(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

class AlexNetNormal(nn.Module):
    def __init__(self, args=None, in_channels = 3, num_classes = 10, norm_type='bn'):
        super(AlexNetNormal, self).__init__()

        # torch.Size([64, 3, 5, 5])  75
        # torch.Size([192, 64, 5, 5]) 1600
        # torch.Size([384, 192, 3, 3]) 1728
        # torch.Size([256, 384, 3, 3]) 3456
        # torch.Size([256, 256, 3, 3]) 2304

        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            ConvBlock(192, 384, bn=norm_type),
            ConvBlock(384, 256, bn=norm_type),
            ConvBlock(256, 256, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
        )

        # cifar10 换成4*4*256, input换成1
        self.classifier = nn.Linear(4 * 4 * 256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    def __init__(self, args=None, in_channels = 3, num_classes = 10, norm_type='bn'):
        super(VGG, self).__init__()

        # torch.Size([64, 3, 3, 3])       # 27
        # torch.Size([64, 64, 3, 3])      # 576 features.1.conv.weight
        # torch.Size([128, 64, 3, 3])     # 576 features.3.conv.weight
        # torch.Size([128, 128, 3, 3])    # 1152 features.4.conv.weight
        # torch.Size([256, 128, 3, 3])    # 1152 features.6.conv.weight
        # torch.Size([256, 256, 3, 3])    # 2304 features.7.conv.weight
        # torch.Size([512, 256, 3, 3])    # 2304 features.9.conv.weight
        # torch.Size([512, 512, 3, 3])    # 4608 features.10.conv.weight
        # torch.Size([512, 512, 3, 3])    # 4608 features.12.conv.weight
        # torch.Size([512, 512, 3, 3])    # 4608 features.13.conv.weight

        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, 3, 1, 1, bn=norm_type),
            ConvBlock(64, 64, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),  # 16x16
            ConvBlock(64, 128, 3, 1, 1, bn=norm_type),
            ConvBlock(128, 128, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),  # 8x8
            ConvBlock(128, 256, 3, 1, 1, bn=norm_type),
            ConvBlock(256, 256, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            ConvBlock(256, 512, 3, 1, 1, bn=norm_type),
            ConvBlock(512, 512, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            ConvBlock(512, 512, 3, 1, 1, bn=norm_type),
            ConvBlock(512, 512, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),  # 4x4
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


import math
from addition_experiment.blocks import BaseBlock

class MobileNetV2(nn.Module):
    def __init__(self, args=None, output_size=10, alpha = 1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        # conv0.weight torch.Size([32, 3, 3, 3])                      # 27
        # bottlenecks.0.conv1.weight torch.Size([32, 32, 1, 1])       # 32
        # bottlenecks.0.conv2.weight torch.Size([32, 1, 3, 3])        # 9
        # bottlenecks.0.conv3.weight torch.Size([16, 32, 1, 1])       # 32
        # bottlenecks.1.conv1.weight torch.Size([96, 16, 1, 1])       # 16
        # bottlenecks.1.conv2.weight torch.Size([96, 1, 3, 3])        # 9
        # bottlenecks.1.conv3.weight torch.Size([24, 96, 1, 1])       # 96
        # bottlenecks.2.conv1.weight torch.Size([144, 24, 1, 1])      # 24
        # bottlenecks.2.conv2.weight torch.Size([144, 1, 3, 3])       # 9
        # bottlenecks.2.conv3.weight torch.Size([24, 144, 1, 1])      # 144
        # bottlenecks.3.conv1.weight torch.Size([144, 24, 1, 1])      # 24
        # bottlenecks.3.conv2.weight torch.Size([144, 1, 3, 3])       # 9
        # bottlenecks.3.conv3.weight torch.Size([32, 144, 1, 1])      # 144
        # bottlenecks.4.conv1.weight torch.Size([192, 32, 1, 1])      # 32
        # bottlenecks.4.conv2.weight torch.Size([192, 1, 3, 3])       # 9
        # bottlenecks.4.conv3.weight torch.Size([32, 192, 1, 1])      # 192
        # bottlenecks.5.conv1.weight torch.Size([192, 32, 1, 1])      # 32
        # bottlenecks.5.conv2.weight torch.Size([192, 1, 3, 3])       # 9
        # bottlenecks.5.conv3.weight torch.Size([32, 192, 1, 1])      # 192
        # bottlenecks.6.conv1.weight torch.Size([192, 32, 1, 1])      # 32
        # bottlenecks.6.conv2.weight torch.Size([192, 1, 3, 3])       # 9
        # bottlenecks.6.conv3.weight torch.Size([64, 192, 1, 1])      # 192
        # bottlenecks.7.conv1.weight torch.Size([384, 64, 1, 1])      # 64
        # bottlenecks.7.conv2.weight torch.Size([384, 1, 3, 3])       # 9
        # bottlenecks.7.conv3.weight torch.Size([64, 384, 1, 1])      # 384
        # bottlenecks.8.conv1.weight torch.Size([384, 64, 1, 1])      # 64
        # bottlenecks.8.conv2.weight torch.Size([384, 1, 3, 3])       # 9
        # bottlenecks.8.conv3.weight torch.Size([64, 384, 1, 1])      # 384
        # bottlenecks.9.conv1.weight torch.Size([384, 64, 1, 1])      # 64
        # bottlenecks.9.conv2.weight torch.Size([384, 1, 3, 3])       # 9
        # bottlenecks.9.conv3.weight torch.Size([64, 384, 1, 1])      # 384
        # bottlenecks.10.conv1.weight torch.Size([384, 64, 1, 1])     # 64
        # bottlenecks.10.conv2.weight torch.Size([384, 1, 3, 3])      # 9
        # bottlenecks.10.conv3.weight torch.Size([96, 384, 1, 1])     # 384
        # bottlenecks.11.conv1.weight torch.Size([576, 96, 1, 1])     # 96
        # bottlenecks.11.conv2.weight torch.Size([576, 1, 3, 3])      # 9
        # bottlenecks.11.conv3.weight torch.Size([96, 576, 1, 1])     # 576
        # bottlenecks.12.conv1.weight torch.Size([576, 96, 1, 1])     # 96
        # bottlenecks.12.conv2.weight torch.Size([576, 1, 3, 3])      # 9
        # bottlenecks.12.conv3.weight torch.Size([96, 576, 1, 1])     # 576
        # bottlenecks.13.conv1.weight torch.Size([576, 96, 1, 1])     # 96
        # bottlenecks.13.conv2.weight torch.Size([576, 1, 3, 3])      # 9
        # bottlenecks.13.conv3.weight torch.Size([160, 576, 1, 1])    # 576
        # bottlenecks.14.conv1.weight torch.Size([960, 160, 1, 1])    # 160
        # bottlenecks.14.conv2.weight torch.Size([960, 1, 3, 3])      # 9
        # bottlenecks.14.conv3.weight torch.Size([160, 960, 1, 1])    # 960
        # bottlenecks.15.conv1.weight torch.Size([960, 160, 1, 1])    # 160
        # bottlenecks.15.conv2.weight torch.Size([960, 1, 3, 3])      # 9
        # bottlenecks.15.conv3.weight torch.Size([160, 960, 1, 1])    # 960
        # bottlenecks.16.conv1.weight torch.Size([960, 160, 1, 1])    # 160
        # bottlenecks.16.conv2.weight torch.Size([960, 1, 3, 3])      # 9
        # bottlenecks.16.conv3.weight torch.Size([320, 960, 1, 1])    # 960
        # conv1.weight torch.Size([1280, 320, 1, 1])                  # 320

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False),
            BaseBlock(16, 24, downsample = False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample = False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample = True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample = False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample = True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample = False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

    #     # weights init
    #     self.weights_init()


    # def weights_init(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))

    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


    def forward(self, inputs):

        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x