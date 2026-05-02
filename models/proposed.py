"""
Edge-guided image generation model.

논문: "대형 이미지 데이터셋 구축을 위한 객체 엣지 기반 이미지 생성 기법"
입력: 소스 이미지 객체(3ch) + 참조 이미지 객체(3ch) + 소스 엣지(1ch) = 7ch
출력: 생성 이미지(3ch)

구조: 인코더(다운샘플링) + 병목 + 디코더(업샘플링) + 스킵 커넥션
손실: MSE + 엣지 일치 패널티
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# 논문 Fig.4 인코더 블록
# conv2d → activation → conv2d → batchnorm → activation
# -------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, use_spectral=True):
        super().__init__()

        def _conv(ic, oc, k=3, s=1, p=1):
            c = nn.Conv2d(ic, oc, k, stride=s, padding=p, bias=False)
            return nn.utils.spectral_norm(c) if use_spectral else c

        self.block = nn.Sequential(
            _conv(in_ch, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            _conv(out_ch, out_ch, s=stride),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -------------------------------------------------------------------
# 논문 Fig.4 디코더 블록
# convtransposed2d → activation → convtransposed2d → batchnorm → activation
# -------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_last=False, dropout_p=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
        ]

        if not is_last:
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p > 0:
                layers.append(nn.Dropout2d(dropout_p))
        else:
            layers.append(nn.Tanh())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# -------------------------------------------------------------------
# 병목 블록 (잔차 연결로 학습 안정화)
# -------------------------------------------------------------------
class BottleneckBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x) + x


# -------------------------------------------------------------------
# 제안 모델 메인: EdgeGuidedGenerator
# -------------------------------------------------------------------
class EdgeGuidedGenerator(nn.Module):
    """
    소스 객체, 참조 객체, 소스 엣지를 입력받아 새로운 이미지를 생성하는
    인코더-디코더 기반 생성 모델.

    논문에서 제안한 구조 기반으로 설계:
      - 7채널 입력을 인코더에서 잠재 공간으로 압축
      - 디코더에서 소스 이미지 구조를 복원
      - 스킵 커넥션으로 세부 구조 보존
    """

    def __init__(self, img_size=128, base_ch=64):
        super().__init__()
        self.img_size = img_size

        # 인코더 (다운샘플링)
        # 7ch → 64 → 128 → 256 → 512
        self.enc1 = EncoderBlock(7,           base_ch,     stride=2)
        self.enc2 = EncoderBlock(base_ch,     base_ch * 2, stride=2)
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4, stride=2)
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8, stride=2)

        # 병목
        self.bottleneck = BottleneckBlock(base_ch * 8)

        # 디코더 (업샘플링) — 스킵 커넥션으로 in_ch 2배
        self.dec4 = DecoderBlock(base_ch * 8 + base_ch * 8, base_ch * 4, dropout_p=0.3)
        self.dec3 = DecoderBlock(base_ch * 4 + base_ch * 4, base_ch * 2, dropout_p=0.3)
        self.dec2 = DecoderBlock(base_ch * 2 + base_ch * 2, base_ch)
        self.dec1 = DecoderBlock(base_ch     + base_ch,     3, is_last=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, src, ref, edge):
        """
        Args:
            src  : 소스 이미지 객체  (B, 3, H, W)
            ref  : 참조 이미지 객체  (B, 3, H, W)
            edge : 소스 이미지 엣지  (B, 1, H, W)
        Returns:
            생성 이미지 (B, 3, H, W), 값 범위 [-1, 1]
        """
        x = torch.cat([src, ref, edge], dim=1)  # (B, 7, H, W)

        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 병목
        b = self.bottleneck(e4)

        # 디코더 (스킵 커넥션)
        d4  = self.dec4(torch.cat([b,  e4], dim=1))
        d3  = self.dec3(torch.cat([d4, e3], dim=1))
        d2  = self.dec2(torch.cat([d3, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))

        return out


# -------------------------------------------------------------------
# 손실 함수
# 논문 수식 (1): L = (1/n) * Σ(Yi - yi)^2  [MSE]
# + 구조 보존을 위한 엣지 일치 패널티
# -------------------------------------------------------------------
class GeneratorLoss(nn.Module):
    def __init__(self, edge_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.edge_weight = edge_weight

    def _sobel(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        kx = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred, target):
        pixel_loss = self.mse(pred, target)
        if self.edge_weight > 0:
            edge_loss = self.mse(self._sobel(pred), self._sobel(target))
            return pixel_loss + self.edge_weight * edge_loss
        return pixel_loss


# -------------------------------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = EdgeGuidedGenerator(img_size=128).to(device)

    src  = torch.randn(2, 3, 128, 128).to(device)
    ref  = torch.randn(2, 3, 128, 128).to(device)
    edge = torch.randn(2, 1, 128, 128).to(device)

    out = model(src, ref, edge)
    print(f"output : {out.shape}  range=[{out.min():.3f}, {out.max():.3f}]")
    print(f"params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
