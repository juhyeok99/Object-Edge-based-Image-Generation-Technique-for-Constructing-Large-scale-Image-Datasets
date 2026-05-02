"""
비교 모델: GAN
참고: Goodfellow et al., "Generative Adversarial Nets", NIPS 2014
"""

import torch
import torch.nn as nn


class GANGenerator(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, latent_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, src, ref):
        x = torch.cat([src, ref], dim=1)
        return self.dec(self.enc(x))


class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, img, src, ref):
        return self.net(torch.cat([img, src, ref], dim=1))


class GANLoss(nn.Module):
    def __init__(self, lambda_recon=100.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1  = nn.L1Loss()
        self.lambda_recon = lambda_recon

    def disc_loss(self, real_pred, fake_pred):
        return (self.bce(real_pred, torch.ones_like(real_pred))
                + self.bce(fake_pred, torch.zeros_like(fake_pred))) * 0.5

    def gen_loss(self, fake_pred, generated, target):
        return (self.bce(fake_pred, torch.ones_like(fake_pred))
                + self.lambda_recon * self.l1(generated, target))


if __name__ == '__main__':
    G = GANGenerator()
    D = GANDiscriminator()
    s = torch.randn(2, 3, 128, 128)
    r = torch.randn(2, 3, 128, 128)
    fake = G(s, r)
    print("G:", fake.shape, "D:", D(fake, s, r).shape)
