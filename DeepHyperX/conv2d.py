# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
import cv2
import matplotlib.pyplot as plt
# utils
import math
import os
import datetime
import numpy as np
import joblib
import collections
import torchvision
from pylab import *

class ConvEtAl(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init.zeros_(m.bias)

    def __init__(self, in_channels):
        super().__init__()
        # 特征提取模块
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3),      # 输出尺寸: [256, H-2, W-2]
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),                                 # 尺寸减半
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),   # 保持尺寸
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),                                 # 尺寸减半
            nn.AdaptiveAvgPool2d(1)                           # 输出固定尺寸[512,1,1]
        )
        self.apply(self.weight_init)

    def forward(self, x):
        x = x.squeeze(dim=1)  # 移除通道维度 [B,1,C,H,W] => [B,C,H,W]
        x = self.conv_block(x)
        return x.flatten(1)    # 展平为[B, 512]

class TESTEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=15):
        super().__init__()
        # 特征提取器
        self.feature_extractor = ConvEtAl(input_channels)
        
        # 分类模块
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes)
        )
        self.apply(self.weight_init)

    def _get_sizes(self):
        """验证特征维度"""
        test_input = torch.randn(1, 1, self.input_channels, 15, 15)
        features = self.feature_extractor(test_input)
        return features.shape[-1]  # 返回特征维度

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)