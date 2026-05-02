"""
비교 모델: Autoencoder
참고: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """소스 + 참조(6ch) → 인코더 → 디코더 → 생성 이미지(3ch)"""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
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
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
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
        z = self.encoder(x)
        return self.decoder(z)


if __name__ == '__main__':
    m = Autoencoder()
    s = torch.randn(2, 3, 128, 128)
    r = torch.randn(2, 3, 128, 128)
    print(m(s, r).shape)
