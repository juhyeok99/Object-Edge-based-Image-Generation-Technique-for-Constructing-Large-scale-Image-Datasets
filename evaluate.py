"""
전체 모델 비교 평가.
논문 Table 1 형식으로 PSNR/SSIM 출력.

사용 예시:
    python evaluate.py --data_dir ./data/dogs --ckpt_dir ./checkpoints
"""

import argparse
from pathlib import Path

import torch

from models import EdgeGuidedGenerator, Autoencoder, GANGenerator, BaselineGenerator
from utils  import build_loaders
from utils.metrics import psnr_tensor, ssim_tensor


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    type=str, required=True)
    p.add_argument('--ckpt_dir',    type=str, default='checkpoints')
    p.add_argument('--img_size',    type=int, default=128)
    p.add_argument('--batch_size',  type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def eval_model(model, loader, device, use_edge=False):
    model.eval()
    psnr_sum = ssim_sum = 0.0
    for batch in loader:
        src  = batch['src'].to(device)
        ref  = batch['ref'].to(device)
        pred = model(src, ref, batch['edge'].to(device)) if use_edge else model(src, ref)
        psnr_sum += psnr_tensor(pred, src)
        ssim_sum += ssim_tensor(pred, src)
    n = len(loader)
    return psnr_sum / n, ssim_sum / n


def load_ckpt(model, path, device, key='model'):
    if not Path(path).exists():
        return False
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt[key])
    return True


def main():
    args   = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, val_loader = build_loaders(
        args.data_dir, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers, val_ratio=0.2
    )

    ckpt_root = Path(args.ckpt_dir)
    results   = {}

    print("\n" + "=" * 52)
    print(f"{'Method':<22}  {'PSNR':>9}  {'SSIM':>10}")
    print("-" * 52)

    rows = [
        ('previous papers[6]', BaselineGenerator(),    'baseline',    False, 'model'),
        ('GAN[10]',            GANGenerator(),          'gan',         False, 'G'),
        ('Autoencoder[11]',    Autoencoder(),           'autoencoder', False, 'model'),
        ('Proposed Technique', EdgeGuidedGenerator(args.img_size), 'proposed', True, 'model'),
    ]

    for label, model, folder, use_edge, key in rows:
        model = model.to(device)
        ckpt_path = ckpt_root / folder / 'best_model.pth'
        if load_ckpt(model, ckpt_path, device, key):
            p, s = eval_model(model, val_loader, device, use_edge)
            print(f"  {label:<20}  {p:>9.5f}  {s:>10.7f}")
            results[label] = (p, s)
        else:
            print(f"  {label:<20}  체크포인트 없음: {ckpt_path}")
        del model

    print("=" * 52)

    if 'Proposed Technique' in results and 'Autoencoder[11]' in results:
        imp = ((results['Proposed Technique'][0] - results['Autoencoder[11]'][0])
               / results['Autoencoder[11]'][0] * 100)
        print(f"\n제안 기법 vs Autoencoder PSNR 향상: +{imp:.1f}%")


if __name__ == '__main__':
    main()
