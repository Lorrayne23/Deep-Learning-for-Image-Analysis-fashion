import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
from torchsummary import summary
import numpy as np
from pytorch_msssim import ssim  # For SSIM metric

from DCGAN import Discriminator, Generator
from SN_GAN import DiscriminatorSGAN


# KID (Kernel Inception Distance)
def polynomial_mmd(x, y, degree=3, gamma=None, coef0=1):
    x = x.reshape(x.size(0), -1)  # Reshape to 2D
    y = y.reshape(y.size(0), -1)  # Reshape to 2D
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    kernel_xx = (gamma * x.mm(x.t()) + coef0) ** degree
    kernel_yy = (gamma * y.mm(y.t()) + coef0) ** degree
    kernel_xy = (gamma * x.mm(y.t()) + coef0) ** degree
    return kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()


def kernel_inception_distance(real_features, generated_features):
    real_features, generated_features = torch.tensor(real_features), torch.tensor(generated_features)
    return polynomial_mmd(real_features, generated_features)


def save_images(epoch, G, device):
    os.makedirs('outputs', exist_ok=True)
    fixed_noise = torch.randn(32, 100, device=device)
    fake_images = G(fixed_noise).detach().cpu()
    save_image(fake_images, f'outputs/epoch_{epoch}.png', nrow=8, normalize=True)


def data_set(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train(dataloader, G, D, optimizerD, optimizerG, criterion, num_epochs, device):
    d_losses = []
    g_losses = []
    SSIM_SCORES = []
    KID_SCORES = []

    for epoch in range(num_epochs):
        d_loss_total = 0.0
        g_loss_total = 0.0

        for i, (idx, _) in enumerate(dataloader):
            real_images = idx.to(device)
            real_labels = torch.ones(idx.size(0), 1, device=device)
            fake_labels = torch.zeros(idx.size(0), 1, device=device)

            ############################
            # Update D network
            ###########################
            D.zero_grad()
            output = D(real_images)
            errD_real = criterion(output, real_labels)
            errD_real.backward()

            noise = torch.randn(idx.size(0), 100, device=device)
            fake = G(noise)
            output = D(fake.detach())
            errD_fake = criterion(output, fake_labels)
            errD_fake.backward()

            optimizerD.step()
            d_loss_total += errD_real.item() + errD_fake.item()

            ############################
            # Update G network
            ###########################
            G.zero_grad()
            output = D(fake)
            errG = criterion(output, real_labels)
            errG.backward()

            optimizerG.step()
            g_loss_total += errG.item()

            # Compute KID
            kid = kernel_inception_distance(real_images.detach(), fake.detach())
            KID_SCORES.append(kid.item())

            # Compute SSIM
            score = ssim(real_images.detach(), fake.detach())
            SSIM_SCORES.append(score.detach().cpu())

        d_losses.append(d_loss_total / len(dataloader))
        g_losses.append(g_loss_total / len(dataloader))

        torch.save(G.state_dict(), f'./checkpoints/G_epoch_{epoch}.pth')
        torch.save(D.state_dict(), f'./checkpoints/D_epoch_{epoch}.pth')

        print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_losses[-1]:.4f}, g_loss: {g_losses[-1]:.4f}')

        if (epoch + 1) in [10, 30, 50]:
            save_images(epoch + 1, G, device)

    return d_losses, g_losses, SSIM_SCORES, KID_SCORES


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of the latent space (noise)")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes for classification (ACGAN only)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DCGAN configuration
    batch_size = 32
    nc = 1  # Number of channels in input image (Grayscale images)
    ngf = 64  # Generator filter size
    ndf = 64  # Discriminator filter size

    # Initialize models
    D = Discriminator(nc, ndf).to(device)
    G = Generator(nc, ngf, args.latent_dim).to(device)

    optimizerD = optim.Adam(D.parameters(), lr=1e-4)
    optimizerG = optim.Adam(G.parameters(), lr=1e-4)

    criterion = nn.BCELoss()

    # Load dataset
    dataloader = data_set(batch_size)

    # Train DCGAN
    train(dataloader, G, D, optimizerD, optimizerG, criterion, args.epochs, device)

    # SN-GAN configuration
    D = DiscriminatorSGAN(nc, ndf).to(device)
    G = Generator(nc, ngf, args.latent_dim).to(device)

    optimizerD = optim.Adam(D.parameters(), lr=5e-4)
    optimizerG = optim.Adam(G.parameters(), lr=5e-4)

    # Train SN-GAN
    train(dataloader, G, D, optimizerD, optimizerG, criterion, args.epochs, device)
