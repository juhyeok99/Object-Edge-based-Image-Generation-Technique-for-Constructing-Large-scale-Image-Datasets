"""
검증용 CNN 분류 모델.
논문 Section IV-3: 생성 이미지 데이터셋이 딥러닝 훈련에서
실제로 유효한지 검증하기 위한 강아지/비강아지 이진 분류기.
"""

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DogClassifier(nn.Module):
    """
    CNN 기반 강아지 분류기.
    입력: (B, 3, 128, 128)
    출력: (B, 1) — BCEWithLogitsLoss 사용 전제
    """

    def __init__(self, img_size=128):
        super().__init__()

        self.features = nn.Sequential(
            ConvBnRelu(3,   32),
            ConvBnRelu(32,  32),
            nn.MaxPool2d(2),        # 128→64
            nn.Dropout2d(0.1),

            ConvBnRelu(32,  64),
            ConvBnRelu(64,  64),
            nn.MaxPool2d(2),        # 64→32
            nn.Dropout2d(0.1),

            ConvBnRelu(64,  128),
            ConvBnRelu(128, 128),
            nn.MaxPool2d(2),        # 32→16
            nn.Dropout2d(0.2),

            ConvBnRelu(128, 256),
            nn.AdaptiveAvgPool2d(4),  # →(B,256,4,4)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x))

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


if __name__ == '__main__':
    m = DogClassifier()
    x = torch.randn(4, 3, 128, 128)
    print(f"output: {m(x).shape}")
    print(f"params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
