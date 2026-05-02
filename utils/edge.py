"""
Sobel 엣지 추출 모듈.

논문 Section III-2: Gradient 기반 Sobel 알고리즘으로
소스 이미지 객체에서 엣지 성분을 추출한다.

참고: Gao et al., "An improved Sobel edge detection", ICCSIT 2010
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sobel_edge_cv2(img_bgr, ksize=3, blur=True):
    """
    OpenCV 기반 Sobel 엣지 추출.

    Args:
        img_bgr : BGR 이미지 (H, W, 3), uint8
        ksize   : 소벨 커널 크기
        blur    : 노이즈 감소용 가우시안 블러 여부
    Returns:
        edge    : 엣지 이미지 (H, W), uint8
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge = np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    return edge


def sobel_edge_tensor(img_tensor):
    """
    PyTorch 텐서 기반 Sobel 엣지 추출 (미분 가능).
    학습 중 on-the-fly 처리에 사용.

    Args:
        img_tensor : (B, 3, H, W) or (3, H, W), float, [-1,1] 또는 [0,1]
    Returns:
        edge       : (B, 1, H, W) or (1, H, W), float, [-1,1]
    """
    squeeze = False
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        squeeze = True

    gray = (0.299 * img_tensor[:, 0:1]
            + 0.587 * img_tensor[:, 1:2]
            + 0.114 * img_tensor[:, 2:3])

    device, dtype = img_tensor.device, img_tensor.dtype

    kx = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        dtype=dtype, device=device
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        dtype=dtype, device=device
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)

    edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-8)
    edge = edge * 2.0 - 1.0  # [-1, 1]

    if squeeze:
        edge = edge.squeeze(0)
    return edge


def prepare_edge_from_path(img_path, target_size=(128, 128)):
    """이미지 경로 → 엣지 텐서 (1, H, W), float32, [-1,1]"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {img_path}")

    img = cv2.resize(img, target_size)
    edge = sobel_edge_cv2(img)
    edge_t = torch.from_numpy(edge).float() / 255.0
    edge_t = edge_t * 2.0 - 1.0
    return edge_t.unsqueeze(0)


if __name__ == '__main__':
    dummy = torch.randn(2, 3, 128, 128)
    out = sobel_edge_tensor(dummy)
    print(f"shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]")
