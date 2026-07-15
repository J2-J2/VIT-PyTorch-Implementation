# ViT-PyTorch-Implementation

PyTorch를 활용하여 ViT(Vision Transformer)를 밑바닥부터 구현하고 실험한 저장소입니다.

본 프로젝트는 ViT 논문인 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale의 구조를 기반으로 하며, 핵심 구조인 Patch Embedding, Position Embedding, Multi-Head Attention, Feed-Forward Network, Pretraining/Fine-tuning Head를 직접 구현하는 것을 목표로 했습니다.

실험 환경은 `uv`를 통해 관리했으며, Imagenette 데이터셋을 활용하여 구현한 ViT 모델의 학습 및 분류 파이프라인을 검증했습니다.

## Blog Review
본 구현에 대한 구현 과정은 아래 블로그에 정리해 두었습니다.

<p align="center">
  <a href="https://velog.io/@jeongjae/ViT-PyTorch-Implementation">
    <img src="./pinggu_circle.png" width="150" height="150">
  </a>
</p>
https://velog.io/@jeongjae/ViT-PyTorch-Implementation
<p align="center">
  <a href="https://velog.io/@jeongjae/ViT-PyTorch-Implementation">
    <b>[ViT 구현하면서 배운 것들 by J2-J2]</b>
  </a>
</p>

## Environment Setup
```
uv sync
```

## Project Structure
```
.
├── configs/               # 모델 및 학습 하이퍼파라미터 설정 파일
│   ├── VIT_BASE.yaml      # ViT-Base 규모 설정
│   └── vit_custom.yaml    # 커스텀 규모 설정
├── models/                # ViT 모델 구성 코드
│   ├── MHAttention.py     # Multi-Head Attention 구현
│   ├── FFN.py             # Feed-Forward Network 구현
│   ├── Encoder.py         # Encoder Block 구현
│   └── VIT.py             # ViT 전체 모델 구조 구현
├── data/                  # 데이터셋 로딩 코드
│   └── custom_dataset.py  # CustomDataset 구현
├── utils/                 # 학습 및 검증 유틸리티 코드
│   ├── trainer.py         # 학습 루프 및 검증 로직
│   └── metrics.py         # precision/recall/f1 등 평가 지표
├── notebooks/             # Colab 학습 노트북
│   └── train_model.ipynb
├── main.py                # 학습 실행 스크립트
├── pyproject.toml         # 프로젝트 의존성 및 환경 설정
└── README.md
```

## Training
```
python main.py
```

## Reference
* Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021.
