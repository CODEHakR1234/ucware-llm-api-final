#!/bin/bash

echo "🚀 GPU 가속 환경 설정 시작..."

# CUDA 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# GPU 메모리 최적화 설정
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Docling GPU 가속 설정
export DOCLING_DEVICE=cuda
export DOCLING_BATCH_SIZE=4
export DOCLING_USE_FLASH_ATTENTION=true

echo "✅ GPU 환경 변수 설정 완료"
echo "   - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   - DOCLING_DEVICE: $DOCLING_DEVICE"
echo "   - DOCLING_BATCH_SIZE: $DOCLING_BATCH_SIZE"

# PyTorch GPU 확인
python -c "
import torch
if torch.cuda.is_available():
    print(f'🎯 GPU 사용 가능: {torch.cuda.get_device_name(0)}')
    print(f'   메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('⚠️  GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다.')
"

echo "🔧 의존성 설치 중..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate

echo "✅ GPU 가속 설정 완료!" 