#!/usr/bin/env bash
# ⇢ PaliGemma-3B 를 Text-Generation-Inference(TGI)로 띄우는 스크립트
set -euo pipefail

###############################################################################
# 0. 사용자 입력 (기본값: GPU 1 / 포트 8080)
###############################################################################
read -p "🖥️  PaliGemma를 돌릴 GPU 번호(기본 1): " GPU_PALI
GPU_PALI=${GPU_PALI:-1}

read -p "🌐 TGI 노출 포트 번호(기본 8080): " PORT_PALI
PORT_PALI=${PORT_PALI:-8080}

###############################################################################
# 1. 경로 및 변수
###############################################################################
MODEL_NAME="google/paligemma-3b"
BASE_DIR="$HOME/tgi-data-storage"
VENV_DIR="$BASE_DIR/tgi-venv"
HF_CACHE="$BASE_DIR/huggingface-cache"
TGI_CACHE="$BASE_DIR/tgi-cache"
LOG_FILE="tgi.log"

mkdir -p "$BASE_DIR" "$HF_CACHE" "$TGI_CACHE"

###############################################################################
# 2. Python 가상환경
###############################################################################
[[ -d "$VENV_DIR" ]] || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
VENV_PY="$VENV_DIR/bin/python"
echo "[TGI venv] $($VENV_PY -V)"

###############################################################################
# 3. TGI 설치(없으면)
###############################################################################
if ! $VENV_PY - <<'PY' 2>/dev/null
import importlib, sys; sys.exit(0 if importlib.util.find_spec("text_generation_server") else 1)
PY
then
  echo "[TGI] 패키지 설치 중…"
  $VENV_PY -m pip install --upgrade pip wheel
  $VENV_PY -m pip install "text-generation-inference==1.4.3"
fi

###############################################################################
# 4. HF 토큰 입력
###############################################################################
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  read -s -p "🔑  HUGGING_FACE_HUB_TOKEN: " HUGGING_FACE_HUB_TOKEN; echo
  export HUGGING_FACE_HUB_TOKEN
fi
export HF_HOME="$HF_CACHE"  XDG_CACHE_HOME="$TGI_CACHE"

###############################################################################
# 5. 서버 기동
###############################################################################
export CUDA_VISIBLE_DEVICES="$GPU_PALI"

echo -e "\n[🚀] TGI(PaliGemma) → GPU $GPU_PALI / PORT $PORT_PALI"
nohup $VENV_PY -m text_generation_server.cli \
      --model-id  "$MODEL_NAME" \
      --port      "$PORT_PALI" \
      --hostname  "0.0.0.0" \
      --quantize  "bnb.int4" \
      --num-shard 1 \
      > "$LOG_FILE" 2>&1 &

PID=$!
printf "[⌛] PID %s – 로딩 중…\n" "$PID"

for _ in {1..150}; do
  sleep 2
  lsof -i :"$PORT_PALI" &>/dev/null && { echo "✅ Ready! tail -f $LOG_FILE"; exit 0; }
done
echo "❌ 300초 내에 시작되지 않았습니다. tail -f $LOG_FILE 로 확인하세요."
exit 1

