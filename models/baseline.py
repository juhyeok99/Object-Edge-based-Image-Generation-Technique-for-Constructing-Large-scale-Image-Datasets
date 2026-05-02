"""
선행 연구 기반 베이스라인 [6].
엣지 성분 없이 소스+참조 바운딩 박스만 사용.

참고: Lee & Kim, "Synthetic data generation technique using
      object bounding box and original image combination", KIPS 2023
"""

import torch
import torch.nn as nn


class BaselineGenerator(nn.Module):
    """
    논문 [6] 재현: 6채널 입력(소스3 + 참조3) → 인코더 → 디코더 → 생성 이미지(3채널)
    엣지 성분이 없어 구조 정보가 부족 → 글리치 발생 가능
    """

    def __init__(self):
        super().__init__()

        self.enc_layers = nn.ModuleList([
            self._enc_block(6,   64),
            self._enc_block(64,  128),
            self._enc_block(128, 256),
            self._enc_block(256, 512),
        ])

        self.dec_layers = nn.ModuleList([
            self._dec_block(512, 256),
            self._dec_block(256, 128),
            self._dec_block(128, 64),
        ])

        self.last = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _enc_block(in_ch, out_ch):
        # 논문 구조: conv → act → conv → BN → act
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def _dec_block(in_ch, out_ch):
        # 논문 구조: convT → act → convT → BN → act
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, src, ref):
        x = torch.cat([src, ref], dim=1)
        for layer in self.enc_layers:
            x = layer(x)
        for layer in self.dec_layers:
            x = layer(x)
        return self.last(x)


if __name__ == '__main__':
    m = BaselineGenerator()
    s = torch.randn(2, 3, 128, 128)
    r = torch.randn(2, 3, 128, 128)
    print(m(s, r).shape)
