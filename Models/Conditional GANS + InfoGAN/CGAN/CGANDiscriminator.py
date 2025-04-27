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


class Discriminator(nn.Module):
    def __init__(self, nz=1, nc=1, input_size=32, class_num=10):
        super(Discriminator, self).__init__()
        self.nz = nz
        self.nc = nc
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.nz + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.nc),
            nn.Sigmoid(),
        )

        # Define weights

        def initialize_weights(net):
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.02)
                    m.bias.data.zero_()
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0, 0.02)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.02)
                    m.bias.data.zero_()
        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x
