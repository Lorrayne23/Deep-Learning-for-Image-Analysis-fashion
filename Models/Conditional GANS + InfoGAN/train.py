import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
from CGAN import CGANGenerator, CGANDiscriminator
from InfoGAN import InfoGANGenerator
from ACGAN import DiscriminatorACGAN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from pytorch_msssim import ssim  # For SSIM metric

# Global loss function
BCE_loss = nn.BCELoss()

def save_images(epoch, y_, model_type):
    os.makedirs('outputs', exist_ok=True)
    z_ = torch.rand((batch_size, z_dim))
    y_vec_ = torch.zeros((batch_size, class_num)) \
            .scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
    fake_images = G(z_, y_vec_).detach().cpu()
    save_image(fake_images, f'outputs/{model_type}{epoch}.png', nrow=8, normalize=True)

def label_processing(batch_size, z_dim, class_num, input_size, y_):
    z_ = torch.rand((batch_size, z_dim))
    y_vec_ = torch.zeros((batch_size, class_num)) \
            .scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)

    y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3) \
              .expand(batch_size, class_num, input_size, input_size)
    return z_, y_vec_, y_fill_

def polynomial_mmd(x, y, degree=3, gamma=None, coef0=1):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dimensions do not match: x has {x.shape[1]} features, y has {y.shape[1]} features.")
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    kernel_xx = (gamma * x.mm(x.t()) + coef0) ** degree
    kernel_yy = (gamma * y.mm(y.t()) + coef0) ** degree
    kernel_xy = (gamma * x.mm(y.t()) + coef0) ** degree
    return kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()

def kernel_inception_distance(real_features, generated_features):
    if real_features.shape[1] != generated_features.shape[1]:
        real_features = real_features[:, :generated_features.shape[1], :, :]
    real_features, generated_features = torch.tensor(real_features), torch.tensor(generated_features)
    return polynomial_mmd(real_features, generated_features)

class Train:

    @staticmethod
    def data_set(batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        return DataLoader(train_dataset, batch_size)

    @staticmethod
    def initialization(sample_num, z_dim, class_num):
        sample_z_ = torch.zeros((sample_num, z_dim))
        for i in range(class_num):
            sample_z_[i * class_num] = torch.rand(1, z_dim)
            for j in range(1, class_num):
                sample_z_[i * class_num + j] = sample_z_[i * class_num]

        temp = torch.zeros((class_num, 1))
        for i in range(class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((sample_num, 1))
        for i in range(class_num):
            temp_y[i * class_num: (i + 1) * class_num] = temp

        sample_y_ = torch.zeros((sample_num, class_num)) \
                    .scatter_(1, temp_y.type(torch.LongTensor), 1)

        return sample_z_, sample_y_

    @staticmethod
    def train_cgan(d_loss_total, g_loss_total, x_, y_fill_, z_, y_vec_, y_real_, y_fake_):
        ############################
        # (1) Update D network
        ###########################
        optimizerD.zero_grad()

        output = D(x_, y_fill_)
        errD_real = BCE_loss(output, y_real_)

        G_ = G(z_, y_vec_)
        D_fake = D(G_, y_fill_)
        errD_fake = BCE_loss(D_fake, y_fake_)

        D_loss = errD_real + errD_fake
        d_loss_total += errD_real.item() + errD_fake.item()

        D_loss.backward()
        optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        optimizerG.zero_grad()

        G_ = G(z_, y_vec_)
        output = D(G_, y_fill_)
        errG = BCE_loss(output, y_real_)

        errG.backward()
        optimizerG.step()

        g_loss_total += errG.item()

        # Calculate KID
        real_images = torch.cat([x_, y_fill_], 1)
        fake = G_

        return D_loss, errG, real_images, fake

    @staticmethod
    def train_model(model_type, G, D, version, sample_num, z_dim, class_num, input_size):
        run = wandb.init(
            entity="lorrayne-reis-silva-city-university-of-london",
            project=f"{model_type}{version}",
            config={
                "learning_rate_G": 0.000055,
                "learning_rate_D": 0.0002,
                "architecture": f"{model_type}{version}",
                "dataset": "Fashion-MNIST",
                "epochs": 20,
            },
        )

        data_loader = Train.data_set(batch_size)
        sample_z_, sample_y_ = Train.initialization(sample_num, z_dim, class_num)
        y_real_, y_fake_ = torch.ones(batch_size, 1), torch.zeros(batch_size, 1)

        d_losses, g_losses, SSIM_SCORES, KID_SCORES = [], [], [], []
        for epoch in range(20):
            d_loss_total = 0.0
            g_loss_total = 0.0

            for iter, (x_, y_) in enumerate(data_loader):
                if iter == data_loader.dataset.__len__() // batch_size:
                    break

                z_, y_vec_, y_fill_ = label_processing(batch_size, z_dim, class_num, input_size, y_)
                x_ = x_.view(x_.size(0), 1, input_size, input_size)

                if model_type in ["CGAN", "CGAN_complete", "InfoGan-CGAN"]:
                    D_loss, errG, real_images, fake = Train.train_cgan(d_loss_total, g_loss_total, x_, y_fill_, z_, y_vec_, y_real_, y_fake_)

                # Calculate KID
                kid = kernel_inception_distance(real_images.detach().cpu(), fake.detach().cpu())

                # Calculate SSIM
                score = ssim(x_, fake.detach())

                if (iter + 1) % 100 == 0:
                    d_losses.append(D_loss.item())
                    g_losses.append(errG.item())
                    run.log({"D_loss": D_loss.item(), "G_loss": errG.item()})
                    KID_SCORES.append(kid.item())
                    SSIM_SCORES.append(score.detach().cpu())

                    print(f"Epoch: [{epoch + 1}] [{iter + 1}/{data_loader.dataset.__len__() // batch_size}] D_loss: {D_loss.item():.8f}, G_loss: {errG.item():.8f}")

            if (epoch + 1) in [1, 10, 15, 20]:
                save_images(epoch + 1, y_, model_type)

            os.makedirs('Epochs', exist_ok=True)
            torch.save(G.state_dict(), f'./Epochs/G_epoch_{model_type}_{epoch}.pth')
            torch.save(D.state_dict(), f'./Epochs/D_epoch_{model_type}_{epoch}.pth')

        run.finish()

        return d_losses, g_losses, SSIM_SCORES, KID_SCORES

if __name__ == "__main__":
    batch_size = 64
    z_dim = 100
    class_num = 10
    input_size = 32  # Adjust based on your dataset or model

    model_type = "CGAN"
    G = CGANGenerator(nz=z_dim, nc=1, input_size=input_size, class_num=class_num)
    D = CGANDiscriminator(nz=1, nc=1, input_size=input_size, class_num=class_num)

    optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    train = Train()
    train.train_model(model_type, G, D, version="v1", sample_num=1000, z_dim=z_dim, class_num=class_num, input_size=input_size)

    model_type = "InfoGan-CGAN"
    G = InfoGANGenerator(nz=z_dim, nc=1, input_size=input_size, class_num=class_num)
    D = CGANDiscriminator(nz=1, nc=1, input_size=input_size, class_num=class_num)
    train.train_model(model_type, G, D, version="v1", sample_num=1000, z_dim=z_dim, class_num=class_num,
                      input_size=input_size)

    model_type = "InfoGan-ACGAN"
    G = InfoGANGenerator(nz=z_dim, nc=1, input_size=input_size, class_num=class_num)
    D = DiscriminatorACGAN(nz=1, nc=1, input_size=input_size, class_num=class_num)
    train.train_model(model_type, G, D, version="v1", sample_num=1000, z_dim=z_dim, class_num=class_num,
                      input_size=input_size)

    model_type = "CGAN_complete"
    G = InfoGANGenerator(nz=z_dim, nc=1, input_size=input_size, class_num=class_num)
    D = DiscriminatorACGAN(nz=1, nc=1, input_size=input_size, class_num=class_num)
    train.train_model(model_type, G, D, version="v1", sample_num=1000, z_dim=z_dim, class_num=class_num,
                      input_size=input_size)

    # Argument parsing for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
