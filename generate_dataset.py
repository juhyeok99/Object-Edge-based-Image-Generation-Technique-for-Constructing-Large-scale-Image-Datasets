"""
이미지 데이터셋 생성 스크립트.

학습된 EdgeGuidedGenerator로 이미지를 생성하고
PSNR/SSIM 필터링 후 저장. (논문 Section IV-2)

사용 예시:
    python generate_dataset.py \\
        --data_dir ./data/dogs \\
        --ckpt checkpoints/proposed/best_model.pth \\
        --out_dir ./data/generated \\
        --n_images 2000
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models import EdgeGuidedGenerator
from utils.edge import sobel_edge_tensor
from utils.metrics import psnr_tensor, ssim_tensor, QualityFilter
from utils.data_utils import to_tensor


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  type=str, required=True)
    p.add_argument('--ckpt',      type=str, required=True)
    p.add_argument('--out_dir',   type=str, default='./data/generated')
    p.add_argument('--n_images',  type=int, default=2000)
    p.add_argument('--img_size',  type=int, default=128)
    p.add_argument('--use_yolo',  action='store_true')
    p.add_argument('--device',    type=str, default=None)
    return p.parse_args()


def tensor_to_bgr(t):
    """(1,3,H,W) 텐서 → BGR uint8"""
    arr = ((t.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
    return cv2.cvtColor(arr.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def get_paths(root):
    return sorted([p for p in Path(root).rglob('*') if p.suffix.lower() in IMG_EXTS])


def main():
    args   = get_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model = EdgeGuidedGenerator(img_size=args.img_size).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"모델 로드: {args.ckpt}")

    detector = None
    if args.use_yolo:
        from utils.detection import ObjectDetector
        detector = ObjectDetector()

    paths = get_paths(args.data_dir)
    if len(paths) < 2:
        raise RuntimeError("이미지 2장 이상 필요")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def load_crop(path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        if detector:
            img, _ = detector.crop_object(img, (args.img_size, args.img_size))
        else:
            img = cv2.resize(img, (args.img_size, args.img_size))
        return to_tensor(img).unsqueeze(0).to(device)

    # 품질 임계값 추정 (샘플 100개)
    qf = QualityFilter()
    print("품질 임계값 추정 중...")
    with torch.no_grad():
        for i in range(min(100, len(paths))):
            src_t = load_crop(paths[i])
            if src_t is None:
                continue
            ref_t = load_crop(random.choice([p for p in paths if p != paths[i]]))
            if ref_t is None:
                continue
            edge_t = sobel_edge_tensor(src_t)
            pred   = model(src_t, ref_t, edge_t)
            qf.update(psnr_tensor(pred, src_t), ssim_tensor(pred, src_t))
    qf.set_thresholds_from_mean()

    # 생성 및 저장
    saved, tried = 0, 0
    pbar = tqdm(total=args.n_images, desc="생성 중")

    with torch.no_grad():
        while saved < args.n_images:
            if tried > len(paths) * 5:
                print("경고: 목표 수량 미달")
                break

            idx    = tried % len(paths)
            src_t  = load_crop(paths[idx])
            tried += 1
            if src_t is None:
                continue

            ref_t = load_crop(random.choice([p for p in paths if p != paths[idx]]))
            if ref_t is None:
                continue

            edge_t = sobel_edge_tensor(src_t)
            pred   = model(src_t, ref_t, edge_t)

            p_val = psnr_tensor(pred, src_t)
            s_val = ssim_tensor(pred, src_t)

            if qf.is_ok(p_val, s_val):
                out_bgr = tensor_to_bgr(pred)
                fname   = Path(args.out_dir) / f"gen_{saved:05d}.jpg"
                cv2.imwrite(str(fname), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                pbar.update(1)

    pbar.close()
    print(f"\n완료: {saved}장 저장 → {args.out_dir}")
    print(f"통과율: {saved / max(tried, 1) * 100:.1f}%")


if __name__ == '__main__':
    main()
