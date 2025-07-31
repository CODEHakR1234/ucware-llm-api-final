#!/bin/bash

echo "🚀 GPU 가속 모드로 API 서버 시작..."

# GPU 환경 설정 로드
source setup_gpu.sh

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# GPU 모니터링 시작
echo "📊 GPU 상태 모니터링 시작..."
nvidia-smi dmon -s pucvmet -d 5 &
GPU_MONITOR_PID=$!

# API 서버 시작
echo "🌐 FastAPI 서버 시작..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 서버 종료 시 GPU 모니터링도 종료
trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT 