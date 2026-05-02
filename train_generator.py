"""
생성 모델 학습 스크립트.

사용 예시:
    python train_generator.py --data_dir ./data/dogs --model proposed --epochs 100
    python train_generator.py --data_dir ./data/dogs --model autoencoder --epochs 100
    python train_generator.py --data_dir ./data/dogs --model gan --epochs 100
    python train_generator.py --data_dir ./data/dogs --model baseline --epochs 100
"""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import (EdgeGuidedGenerator, GeneratorLoss,
                    Autoencoder, GANGenerator, GANDiscriminator, GANLoss,
                    BaselineGenerator)
from utils import build_loaders
from utils.metrics import psnr_tensor, ssim_tensor


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    type=str,   required=True)
    p.add_argument('--save_dir',    type=str,   default='checkpoints')
    p.add_argument('--model',       type=str,   default='proposed',
                   choices=['proposed', 'baseline', 'autoencoder', 'gan'])
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--lr',          type=float, default=2e-4)
    p.add_argument('--img_size',    type=int,   default=128)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--edge_weight', type=float, default=0.1)
    p.add_argument('--use_yolo',    action='store_true')
    p.add_argument('--resume',      type=str,   default=None)
    return p.parse_args()


# -------------------------------------------------------
def train_one_epoch(model, crit, opt, loader, device, use_edge):
    model.train()
    total_loss, total_psnr = 0.0, 0.0

    for batch in loader:
        src  = batch['src'].to(device)
        ref  = batch['ref'].to(device)

        opt.zero_grad()

        if use_edge:
            edge = batch['edge'].to(device)
            pred = model(src, ref, edge)
        else:
            pred = model(src, ref)

        loss = crit(pred, src)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        with torch.no_grad():
            total_psnr += psnr_tensor(pred, src)

    n = len(loader)
    return total_loss / n, total_psnr / n


@torch.no_grad()
def validate(model, crit, loader, device, use_edge):
    model.eval()
    total_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0

    for batch in loader:
        src  = batch['src'].to(device)
        ref  = batch['ref'].to(device)

        if use_edge:
            edge = batch['edge'].to(device)
            pred = model(src, ref, edge)
        else:
            pred = model(src, ref)

        total_loss += crit(pred, src).item()
        total_psnr += psnr_tensor(pred, src)
        total_ssim += ssim_tensor(pred, src)

    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n


# -------------------------------------------------------
def train_gan_epoch(G, D, g_opt, d_opt, loader, loss_fn, device):
    G.train(); D.train()
    g_total, d_total = 0.0, 0.0

    for batch in loader:
        src = batch['src'].to(device)
        ref = batch['ref'].to(device)

        # 판별기
        fake     = G(src, ref).detach()
        d_loss   = loss_fn.disc_loss(D(src, src, ref), D(fake, src, ref))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        # 생성기
        fake   = G(src, ref)
        g_loss = loss_fn.gen_loss(D(fake, src, ref), fake, src)
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        g_total += g_loss.item()
        d_total += d_loss.item()

    n = len(loader)
    return g_total / n, d_total / n


# -------------------------------------------------------
def main():
    args   = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  model={args.model}")

    out_dir = Path(args.save_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = None
    if args.use_yolo:
        from utils.detection import ObjectDetector
        detector = ObjectDetector()

    tr_loader, val_loader = build_loaders(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        detector=detector,
    )

    # ---- 모델 세팅 ----
    use_edge = (args.model == 'proposed')

    if args.model == 'proposed':
        model = EdgeGuidedGenerator(img_size=args.img_size).to(device)
        crit  = GeneratorLoss(edge_weight=args.edge_weight).to(device)
        opt   = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        sch   = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)

    elif args.model == 'baseline':
        model = BaselineGenerator().to(device)
        crit  = torch.nn.MSELoss()
        opt   = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        sch   = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)

    elif args.model == 'autoencoder':
        model = Autoencoder().to(device)
        crit  = torch.nn.MSELoss()
        opt   = optim.Adam(model.parameters(), lr=args.lr)
        sch   = CosineAnnealingLR(opt, args.epochs)

    elif args.model == 'gan':
        G     = GANGenerator().to(device)
        D     = GANDiscriminator().to(device)
        g_opt = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
        g_sch = CosineAnnealingLR(g_opt, args.epochs)
        d_sch = CosineAnnealingLR(d_opt, args.epochs)
        loss_fn = GANLoss()

    # ---- 이어서 학습 ----
    start = 0
    if args.resume:
        ckpt  = torch.load(args.resume, map_location=device)
        if args.model != 'gan':
            model.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['opt'])
        start = ckpt.get('epoch', 0) + 1
        print(f"에폭 {start}부터 재개")

    log = open(out_dir / 'log.csv', 'w')
    log.write('epoch,train_loss,val_loss,val_psnr,val_ssim\n')

    best_psnr = 0.0

    for ep in range(start, args.epochs):
        t0 = time.time()

        if args.model == 'gan':
            tr_g, tr_d = train_gan_epoch(G, D, g_opt, d_opt, tr_loader, loss_fn, device)
            tr_loss = tr_g

            G.eval()
            val_psnr = val_ssim = 0.0
            with torch.no_grad():
                for b in val_loader:
                    s = b['src'].to(device); r = b['ref'].to(device)
                    p = G(s, r)
                    val_psnr += psnr_tensor(p, s)
                    val_ssim += ssim_tensor(p, s)
            val_psnr /= len(val_loader); val_ssim /= len(val_loader)
            val_loss = 0.0
            g_sch.step(); d_sch.step()

        else:
            tr_loss, tr_psnr = train_one_epoch(model, crit, opt, tr_loader, device, use_edge)
            val_loss, val_psnr, val_ssim = validate(model, crit, val_loader, device, use_edge)
            sch.step()

        elapsed = time.time() - t0
        print(f"[{ep+1:3d}/{args.epochs}] "
              f"loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"PSNR={val_psnr:.3f}  SSIM={val_ssim:.4f}  ({elapsed:.1f}s)")

        log.write(f"{ep+1},{tr_loss:.6f},{val_loss:.6f},{val_psnr:.4f},{val_ssim:.4f}\n")
        log.flush()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            state = {'epoch': ep, 'psnr': val_psnr, 'ssim': val_ssim}
            if args.model == 'gan':
                state.update({'G': G.state_dict(), 'D': D.state_dict()})
            else:
                state.update({'model': model.state_dict(), 'opt': opt.state_dict()})
            torch.save(state, out_dir / 'best_model.pth')
            print(f"  → best 저장 (PSNR={best_psnr:.4f})")

        if (ep + 1) % 10 == 0 and args.model != 'gan':
            torch.save({'epoch': ep, 'model': model.state_dict()},
                       out_dir / f'epoch_{ep+1}.pth')

    log.close()
    print(f"\n완료. 최고 PSNR: {best_psnr:.4f}")


if __name__ == '__main__':
    main()
