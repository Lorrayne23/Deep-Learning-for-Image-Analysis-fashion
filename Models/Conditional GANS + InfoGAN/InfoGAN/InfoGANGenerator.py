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

class Generator(nn.Module):
    def __init__(self,nz=100, nc=1, input_size=32, class_num=10):
        super(Generator, self).__init__()
        self.nz = nz
        self.nc = nc
        self.input_size = input_size
        self.class_num = class_num


        def    initialize_weights(net):
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

        self.fc = nn.Sequential(
            nn.Linear(self.nz+ self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.nc, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

    # Define weights

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x