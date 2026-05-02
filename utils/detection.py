"""
객체 탐지 기반 바운딩 박스 추출 모듈.

논문 Section III-1: 객체 탐지 모델로 소스/참조 이미지에서
가장 큰 바운딩 박스 기준으로 크롭한다.
YOLOv5 사용. 탐지 실패 시 전체 이미지 반환.
"""

import cv2
import numpy as np
import torch
from pathlib import Path


class ObjectDetector:
    """YOLOv5 기반 객체 탐지기"""

    def __init__(self, model_name='yolov5s', conf_thresh=0.25, device=None):
        self.conf_thresh = conf_thresh
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load(model_name)

    def _load(self, name):
        try:
            self.model = torch.hub.load(
                'ultralytics/yolov5', name, pretrained=True, trust_repo=True
            )
            self.model.conf = self.conf_thresh
            self.model.to(self.device).eval()
            print(f"[ObjectDetector] {name} 로드 완료")
        except Exception as e:
            print(f"[ObjectDetector] 로드 실패 ({e}) → 전체 이미지 사용")
            self.model = None

    def detect_largest(self, img_bgr):
        """가장 큰 바운딩 박스 반환. 탐지 실패 시 전체 영역."""
        h, w = img_bgr.shape[:2]
        fallback = (0, 0, w, h)

        if self.model is None:
            return fallback

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = self.model(rgb)

        dets = results.xyxy[0].cpu().numpy()
        dets = dets[dets[:, 4] >= self.conf_thresh]

        if len(dets) == 0:
            return fallback

        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        best  = int(np.argmax(areas))
        x1, y1, x2, y2 = dets[best, :4].astype(int)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return fallback

        return (x1, y1, x2, y2)

    def crop_object(self, img_bgr, target_size=(128, 128)):
        """탐지 박스 기준 크롭 + 리사이즈"""
        x1, y1, x2, y2 = self.detect_largest(img_bgr)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img_bgr
        return cv2.resize(crop, target_size), (x1, y1, x2, y2)


def center_crop(img_bgr, target_size=(128, 128)):
    """YOLOv5 없이 중앙 정방형 크롭 (폴백)"""
    h, w  = img_bgr.shape[:2]
    side  = min(h, w)
    cy, cx = h // 2, w // 2
    half  = side // 2
    crop  = img_bgr[cy - half: cy + half, cx - half: cx + half]
    return cv2.resize(crop, target_size)


def load_and_crop(img_path, detector=None, target_size=(128, 128)):
    """
    이미지 경로 → 크롭된 텐서 (3, H, W), float32, [-1,1]
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지 없음: {img_path}")

    if detector is not None:
        cropped, _ = detector.crop_object(img, target_size)
    else:
        cropped = center_crop(img, target_size)

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1) / 127.5 - 1.0


if __name__ == '__main__':
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    out   = center_crop(dummy)
    print(f"center_crop 결과: {out.shape}")
