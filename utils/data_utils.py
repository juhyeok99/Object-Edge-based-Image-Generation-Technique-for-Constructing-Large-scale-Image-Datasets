"""
데이터셋 및 DataLoader 구성.

- DogDataset     : 소스/참조 쌍 데이터셋 (논문 방식과 동일)
- build_loaders  : 학습/검증 DataLoader 반환
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .edge import sobel_edge_tensor


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def to_tensor(img_bgr, size=(128, 128)):
    """BGR numpy → RGB 텐서 (3, H, W), float32, [-1, 1]"""
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1) / 127.5 - 1.0


class DogDataset(Dataset):
    """
    소스/참조 이미지 쌍 데이터셋.

    소스는 순차 선택, 참조는 소스를 제외한 나머지 중 랜덤 선택.
    (논문 Section III 이미지 선정 방식과 동일)
    """

    def __init__(self, root_dir, img_size=128, detector=None, augment=True):
        self.img_size = img_size
        self.detector = detector
        self.augment  = augment

        self.paths = sorted([
            p for p in Path(root_dir).rglob('*')
            if p.suffix.lower() in IMG_EXTS
        ])

        if not self.paths:
            raise RuntimeError(f"이미지 없음: {root_dir}")

        print(f"[DogDataset] {len(self.paths)}개 이미지 ({root_dir})")

    def __len__(self):
        return len(self.paths)

    def _read(self, path):
        img = cv2.imread(str(path))
        if img is None:
            img = cv2.imread(str(self.paths[0]))
        return img

    def _crop(self, img):
        if self.detector is not None:
            cropped, _ = self.detector.crop_object(img, (self.img_size, self.img_size))
        else:
            cropped = cv2.resize(img, (self.img_size, self.img_size))
        return cropped

    def __getitem__(self, idx):
        src_img = self._read(self.paths[idx])

        ref_idx = random.choice([i for i in range(len(self.paths)) if i != idx])
        ref_img = self._read(self.paths[ref_idx])

        src_t = to_tensor(self._crop(src_img))
        ref_t = to_tensor(self._crop(ref_img))

        # 가로 flip 증강
        if self.augment and random.random() > 0.5:
            src_t = torch.flip(src_t, dims=[-1])
        if self.augment and random.random() > 0.5:
            ref_t = torch.flip(ref_t, dims=[-1])

        edge_t = sobel_edge_tensor(src_t)  # (1, H, W)

        return {
            'src' : src_t,
            'ref' : ref_t,
            'edge': edge_t,
            'src_path': str(self.paths[idx]),
        }


def build_loaders(root_dir, img_size=128, batch_size=16,
                  val_ratio=0.1, num_workers=4, detector=None):
    """학습/검증 DataLoader 반환"""
    ds    = DogDataset(root_dir, img_size=img_size, detector=detector)
    n_val = max(1, int(len(ds) * val_ratio))
    n_tr  = len(ds) - n_val

    tr_ds, val_ds = torch.utils.data.random_split(
        ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    tr_loader  = DataLoader(tr_ds,  batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"[DataLoader] train={len(tr_ds)}, val={len(val_ds)}")
    return tr_loader, val_loader
