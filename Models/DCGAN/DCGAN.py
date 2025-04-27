import torch
import torch.nn.functional as F
from torchvision import models
from pytorch_msssim import ssim  # For SSIM metric
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torchsummary import summary
import torch.optim as optim
import logging



class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.model = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 5, stride=2, padding=2),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.Conv2d(self.ndf,self.ndf*2, 5, stride=2, padding=2),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear((self.ndf*2)*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Baseline from LAB 5 (INM705 Deep Learning for Image Analysis)

class Generator(nn.Module):
      def __init__(self, nc, ngf, nz):
        super(Generator, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.model = nn.Sequential(
            nn.Linear(100, 7*7*(self.ngf*4), bias=False),
            nn.BatchNorm1d(7*7*(self.ngf*4)),
            nn.LeakyReLU(0.3),
            nn.Unflatten(self.nc, (self.ngf*4, 7, 7)),
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(self.ngf*2, self.ngf, 5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(self.ngf, self.nc, 5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )

      def forward(self, x):
        return self.model(x)