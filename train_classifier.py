"""
분류 모델 학습 스크립트.

논문 Section IV-3 실험 재현:
  a) source  : 소스 이미지만
  b) generated: 생성 이미지만
  c) mixed   : 소스 + 생성 혼합

사용 예시:
    python train_classifier.py --source_dir ./data/dogs --neg_dir ./data/cats --mode source
    python train_classifier.py --source_dir ./data/dogs --neg_dir ./data/cats \\
        --generated_dir ./data/generated --mode mixed
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

from classifier import DogClassifier
from utils.data_utils import to_tensor


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


class SimpleDataset(Dataset):
    def __init__(self, img_dir, label, img_size=128, augment=False):
        self.paths    = sorted([p for p in Path(img_dir).rglob('*')
                                if p.suffix.lower() in IMG_EXTS])
        self.label    = float(label)
        self.img_size = img_size
        self.augment  = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        t = to_tensor(img, (self.img_size, self.img_size))
        if self.augment and torch.rand(1) > 0.5:
            t = torch.flip(t, dims=[-1])
        return t, torch.tensor(self.label)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source_dir',    type=str, required=True)
    p.add_argument('--neg_dir',       type=str, required=True)
    p.add_argument('--generated_dir', type=str, default=None)
    p.add_argument('--mode', type=str, default='source',
                   choices=['source', 'generated', 'mixed'])
    p.add_argument('--save_dir',    type=str,   default='checkpoints/classifier')
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--img_size',    type=int,   default=128)
    p.add_argument('--num_workers', type=int,   default=4)
    return p.parse_args()


def build_dataset(args):
    neg_ds = SimpleDataset(args.neg_dir, 0, args.img_size)

    if args.mode == 'source':
        pos_ds = SimpleDataset(args.source_dir, 1, args.img_size, augment=True)

    elif args.mode == 'generated':
        if not args.generated_dir:
            raise ValueError("--generated_dir 필요")
        pos_ds = SimpleDataset(args.generated_dir, 1, args.img_size)

    else:  # mixed
        if not args.generated_dir:
            raise ValueError("--generated_dir 필요")
        pos_ds = ConcatDataset([
            SimpleDataset(args.source_dir,    1, args.img_size, augment=True),
            SimpleDataset(args.generated_dir, 1, args.img_size),
        ])

    ds = ConcatDataset([pos_ds, neg_ds])
    print(f"[{args.mode}] 전체 {len(ds)}개")
    return ds


def main():
    args   = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = Path(args.save_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    ds    = build_dataset(args)
    n_val = max(1, int(len(ds) * 0.15))
    tr_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                  generator=torch.Generator().manual_seed(0))

    tr_loader  = DataLoader(tr_ds,  args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = DogClassifier(img_size=args.img_size).to(device)
    crit  = nn.BCEWithLogitsLoss()
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch   = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)

    best_acc = 0.0
    log = open(out_dir / 'log.csv', 'w')
    log.write('epoch,tr_loss,val_loss,val_acc\n')

    for ep in range(args.epochs):
        # 학습
        model.train()
        tr_loss = tr_c = tr_n = 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out  = model(imgs).squeeze(1)
            loss = crit(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
            tr_c    += ((torch.sigmoid(out) > 0.5).long() == labels.long()).sum().item()
            tr_n    += labels.size(0)
        sch.step()

        # 검증
        model.eval()
        val_loss = val_c = val_n = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs).squeeze(1)
                val_loss += crit(out, labels).item()
                val_c    += ((torch.sigmoid(out) > 0.5).long() == labels.long()).sum().item()
                val_n    += labels.size(0)

        tr_acc  = tr_c  / tr_n
        val_acc = val_c / val_n
        tl = tr_loss  / len(tr_loader)
        vl = val_loss / len(val_loader)

        print(f"[{ep+1:3d}/{args.epochs}] "
              f"loss={tl:.4f}  val_loss={vl:.4f}  "
              f"acc={tr_acc:.4f}  val_acc={val_acc:.4f}")
        log.write(f"{ep+1},{tl:.6f},{vl:.6f},{val_acc:.6f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': ep, 'model': model.state_dict(),
                        'val_acc': val_acc}, out_dir / 'best_model.pth')

    log.close()
    print(f"\n[{args.mode}] 최고 val_acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()
