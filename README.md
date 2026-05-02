# Object Edge-based Image Generation

논문 구현: **"대형 이미지 데이터셋 구축을 위한 객체 엣지 기반 이미지 생성 기법"**  
이주혁, 김미희 (한경국립대학교), IKEEE Vol.27 No.3, 2023

---

## 디렉토리 구조

```
edge_image_gen/
├── models/
│   ├── __init__.py
│   ├── proposed.py       # 제안 모델 (EdgeGuidedGenerator)
│   ├── autoencoder.py    # 비교 모델
│   ├── gan.py            # 비교 모델
│   └── baseline.py       # 선행 연구 [6]
├── utils/
│   ├── __init__.py
│   ├── edge.py           # Sobel 엣지 추출
│   ├── detection.py      # YOLOv5 객체 탐지
│   ├── metrics.py        # PSNR, SSIM
│   └── data_utils.py     # 데이터셋 로더
├── classifier/
│   ├── __init__.py
│   └── cnn.py            # 검증용 CNN 분류기
├── data/
│   └── dogs/             # 학습 이미지 위치
├── checkpoints/          # 모델 저장 위치
├── train_generator.py    # 생성 모델 학습
├── train_classifier.py   # 분류 모델 학습
├── evaluate.py           # Table 1 형식 평가
├── generate_dataset.py   # 데이터셋 생성
└── requirements.txt
```

---

## 설치

```bash
pip install -r requirements.txt
```

데이터: [Kaggle Dogs vs Cats](https://www.kaggle.com/datasets/tongpython/cat-and-dog)  
강아지 이미지를 `./data/dogs/` 에 넣어주세요.

---

## 사용법

### 1. 생성 모델 학습

```bash
# 제안 기법 (논문 구현)
python train_generator.py --data_dir ./data/dogs --model proposed --epochs 100

# 비교 모델
python train_generator.py --data_dir ./data/dogs --model autoencoder --epochs 100
python train_generator.py --data_dir ./data/dogs --model gan --epochs 100
python train_generator.py --data_dir ./data/dogs --model baseline --epochs 100

# YOLOv5 객체 탐지 사용 시
python train_generator.py --data_dir ./data/dogs --model proposed --use_yolo
```

### 2. 이미지 데이터셋 생성

```bash
python generate_dataset.py \
    --data_dir ./data/dogs \
    --ckpt checkpoints/proposed/best_model.pth \
    --out_dir ./data/generated \
    --n_images 2000
```

### 3. 모델 평가 (PSNR / SSIM)

```bash
python evaluate.py --data_dir ./data/dogs --ckpt_dir ./checkpoints
```

### 4. 분류 모델로 데이터셋 유효성 검증 (논문 Section IV-3)

```bash
# a) 소스 이미지만
python train_classifier.py \
    --source_dir ./data/dogs --neg_dir ./data/cats --mode source

# b) 생성 이미지만
python train_classifier.py \
    --source_dir ./data/dogs --neg_dir ./data/cats \
    --generated_dir ./data/generated --mode generated

# c) 소스 + 생성 혼합 (논문 권장)
python train_classifier.py \
    --source_dir ./data/dogs --neg_dir ./data/cats \
    --generated_dir ./data/generated --mode mixed
```

---

## 논문 재현 결과 (Table 1)

| Method | PSNR | SSIM |
|:---|---:|---:|
| previous papers[6] | 14.433 | 0.4939 |
| GAN[10] | 26.243 | 0.8547 |
| Autoencoder[11] | 23.134 | 0.8313 |
| **Proposed Technique** | **31.249** | **0.9353** |

---

## 제안 모델 구조

입력 7채널(소스 3 + 참조 3 + 엣지 1) → 인코더 4단계 → 병목 → 디코더 4단계 → 출력 3채널

- 인코더 블록: `Conv2d → LeakyReLU → Conv2d → BatchNorm2d → LeakyReLU`
- 디코더 블록: `ConvTranspose2d → ReLU → ConvTranspose2d → BatchNorm2d → ReLU`
- 스킵 커넥션(U-Net 방식), Spectral Normalization, Residual Bottleneck 적용
- 손실 함수: MSE + Sobel 엣지 일치 패널티
