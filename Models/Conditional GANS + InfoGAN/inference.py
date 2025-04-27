import torch
import matplotlib.pyplot as plt
from InfoGAN import InfoGANGenerator
from CGAN import CGANGenerator

def show_image(tensor):
    image = tensor.view(28, 28).detach().cpu().numpy()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def inference(checkpoint_path, latent_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for Generator in (InfoGANGenerator,CGANGenerator):
        G = Generator(latent_dim)
        G.load_state_dict(torch.load(checkpoint_path, map_location=device))
        G.eval()

        z = torch.randn(1, latent_dim)
        fake_img = G(z)
        show_image(fake_img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/Generator.th")
    parser.add_argument("--latent_dim", type=int, default=100)
    args = parser.parse_args()

    inference(args.checkpoint, args.latent_dim)
