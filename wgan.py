import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="dataset/")
args = parser.parse_args()

device = "cuda"
LEARNING_RATE = 1e-4
BATCH_SIZE = 72
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 30
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
GEN_EVERY = 100
SAVE_EVERY_EPOCH = 10
NUM_WORKERS = 16
manualSeed = 999

random.seed(manualSeed)
torch.manual_seed(manualSeed)

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.ImageFolder(root=args.path, transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 64, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success")


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def main():
    test()
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # for tensorboard plotting
    fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(device)

    # get date as filename
    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer_real = SummaryWriter(f"logs/WGAN_celeba/{datestr}/real")
    writer_fake = SummaryWriter(f"logs/WGAN_celeba/{datestr}/fake")
    writer_losses = SummaryWriter(f"logs/WGAN_celeba/{datestr}/losses")

    step = 0
    every_n_critic = 0

    gen.train()
    critic.train()
    loss_gen = 0

    fake_img_list = []

    for epoch in tqdm(range(NUM_EPOCHS)):
        for batch_idx, (real, _) in enumerate(tqdm(loader, leave=False)):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            if batch_idx % CRITIC_ITERATIONS == 0:
                gen.zero_grad()
                loss_gen = -torch.mean(critic(fake).reshape(-1))
                loss_gen.backward()
                opt_gen.step()
                writer_losses.add_scalars(
                    "losses",
                    {"critic_loss": loss_critic, "gen_loss": loss_gen},
                    every_n_critic,
                )
                every_n_critic+=1

            if batch_idx % GEN_EVERY == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)

                    img_grid_real = torchvision.utils.make_grid(
                        real[:64], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:64], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    fake_img_list.append(img_grid_fake.cpu().numpy())

                step += 1
                
        if(epoch % SAVE_EVERY_EPOCH == 0):
            torch.save(critic.state_dict(), f"wgan_critic-{epoch}.pth")
            torch.save(gen.state_dict(), f"wgan_gen-{epoch}.pth")
            

    torch.save(critic.state_dict(), f"wgan_critic-{NUM_EPOCHS}.pth")
    torch.save(gen.state_dict(), f"wgan_gen-{NUM_EPOCHS}.pth")

    ani_fig = plt.figure()
    plt.axis("off")
    ims = [
        [plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in fake_img_list
    ]
    ani = animation.ArtistAnimation(
        ani_fig, ims, interval=1000, repeat_delay=1000, blit=True
    )
    ani.save(f"wgan_celeba-{NUM_EPOCHS}.gif", fps=60)


if __name__ == "__main__":
    main()
