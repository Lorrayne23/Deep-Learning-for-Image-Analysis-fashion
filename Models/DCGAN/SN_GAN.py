
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
import utils

class DiscriminatorSGAN(nn.Module):
    def __init__(self, nc, ndf):
        super(DiscriminatorSGAN, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(self.nc, self.ndf, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.3),
            utils.spectral_norm(nn.Conv2d(self.ndf,self.ndf*2, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.3),
            nn.Flatten(),
            utils.spectral_norm(nn.Linear((self.ndf*2)*7*7, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)