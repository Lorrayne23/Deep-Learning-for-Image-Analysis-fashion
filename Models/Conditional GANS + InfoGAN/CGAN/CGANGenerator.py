from torchvision import datasets, transforms

from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torchsummary import summary
import torch.optim as optim
import logging

import wandb
import random

import torch, time, os, pickle
import os
import torch
import numpy as np

from os import listdir

import torch
import torch.nn.functional as F

from pytorch_msssim import ssim  # For SSIM metric
import numpy as np
import torch.nn as nn
import torch.optim as optim

import matplotlib.image as mpimg


class GeneratorCGAN(nn.Module):
    def __init__(self, nz=100, nc=1, input_size=32, class_num=10):
        super(GeneratorCGAN, self).__init__()
        self.nz = nz
        self.nc = nc
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.nz + self.class_num, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.nc, 4, 2, 1),
            nn.Tanh(),
        )

        self.apply(self._initialize_weights)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)

        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))

        x = self.deconv(x)

        return x

    def _initialize_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)