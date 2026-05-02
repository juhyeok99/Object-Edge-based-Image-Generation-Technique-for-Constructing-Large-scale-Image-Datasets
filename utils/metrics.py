"""
이미지 품질 평가 지표: PSNR, SSIM

논문 Section IV-2:
  - PSNR : 픽셀 강도 차이 측정
  - SSIM : 구조적/지각적 유사도 측정

논문 Table 1 기준값:
  - 제안 기법: PSNR=31.249, SSIM=0.935
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def psnr_tensor(pred, target):
    """
    텐서 PSNR. 입력 범위 [-1, 1] → max_val=2.0

    Returns:
        스칼라 float
    """
    mse = F.mse_loss(pred.detach(), target.detach())
    if mse < 1e-10:
        return 100.0
    return float(20.0 * torch.log10(torch.tensor(2.0) / torch.sqrt(mse)))


def psnr_numpy(img1, img2, max_val=255.0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse  = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def _gauss_kernel(win=11, sigma=1.5, ch=3):
    coords = torch.arange(win, dtype=torch.float32) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = g.unsqueeze(0) * g.unsqueeze(1)
    return k2d.unsqueeze(0).unsqueeze(0).expand(ch, 1, win, win).contiguous()


def ssim_tensor(pred, target, win=11, C1=1e-4, C2=9e-4):
    """
    SSIM. 입력 [-1,1] → 내부에서 [0,1]로 변환.

    Returns:
        스칼라 float
    """
    pred   = (pred   + 1.0) / 2.0
    target = (target + 1.0) / 2.0

    ch     = pred.shape[1]
    kernel = _gauss_kernel(win, 1.5, ch).to(pred.device)
    pad    = win // 2

    mu1 = F.conv2d(pred,   kernel, padding=pad, groups=ch)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=ch)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    s1 = F.conv2d(pred   * pred,   kernel, padding=pad, groups=ch) - mu1_sq
    s2 = F.conv2d(target * target, kernel, padding=pad, groups=ch) - mu2_sq
    s12= F.conv2d(pred   * target, kernel, padding=pad, groups=ch) - mu12

    num = (2 * mu12 + C1) * (2 * s12  + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)

    return float((num / den).mean())


def ssim_numpy(img1, img2):
    t1 = torch.from_numpy(img1.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    t2 = torch.from_numpy(img2.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return ssim_tensor(t1, t2)


class QualityFilter:
    """
    PSNR/SSIM 임계값 기반 이미지 품질 필터.
    논문: 평균값 이상인 이미지만 데이터셋에 포함.
    """

    def __init__(self, psnr_thresh=None, ssim_thresh=None):
        self.psnr_thresh = psnr_thresh
        self.ssim_thresh = ssim_thresh
        self._psnr_buf   = []
        self._ssim_buf   = []

    def update(self, psnr_val, ssim_val):
        self._psnr_buf.append(psnr_val)
        self._ssim_buf.append(ssim_val)

    def set_thresholds_from_mean(self):
        self.psnr_thresh = float(np.mean(self._psnr_buf))
        self.ssim_thresh = float(np.mean(self._ssim_buf))
        print(f"[QualityFilter] PSNR 임계값={self.psnr_thresh:.4f}, "
              f"SSIM 임계값={self.ssim_thresh:.4f}")

    def is_ok(self, psnr_val, ssim_val):
        p_ok = self.psnr_thresh is None or psnr_val >= self.psnr_thresh
        s_ok = self.ssim_thresh is None or ssim_val >= self.ssim_thresh
        return p_ok and s_ok

    @property
    def mean_psnr(self):
        return float(np.mean(self._psnr_buf)) if self._psnr_buf else 0.0

    @property
    def mean_ssim(self):
        return float(np.mean(self._ssim_buf)) if self._ssim_buf else 0.0


if __name__ == '__main__':
    a = torch.rand(2, 3, 128, 128)
    b = a + torch.randn_like(a) * 0.05
    print(f"PSNR: {psnr_tensor(a, b):.4f}")
    print(f"SSIM: {ssim_tensor(a, b):.4f}")
