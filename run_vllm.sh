#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 경로 설정
###############################################################################
BASE_DIR="$HOME/vllm-data-storage"
VENV_DIR="$BASE_DIR/vllm-venv"           # 새·기존 venv 유지
HF_CACHE="$BASE_DIR/huggingface-cache"
VLLM_CACHE="$BASE_DIR/vllm-cache"
LOG_FILE="vllm.log"
PORT=12000
MODEL_NAME="Qwen/Qwen3-30B-A3B-GPTQ-Int4"

mkdir -p "$BASE_DIR" "$HF_CACHE" "$VLLM_CACHE"

###############################################################################
# 1. 가상환경
###############################################################################
[[ -d "$VENV_DIR" ]] || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
VENV_PY="$VENV_DIR/bin/python"
echo "[venv] $($VENV_PY -V)"

###############################################################################
# 2. vLLM 설치 (cu118 pre-built 휠)
###############################################################################
if ! $VENV_PY - <<'PY' 2>/dev/null
import importlib, sys; sys.exit(0 if importlib.util.find_spec("vllm") else 1)
PY
then
  echo "[vLLM] cu118 사전 휠 설치 중…"
  $VENV_PY -m pip install --upgrade pip wheel
  $VENV_PY -m pip install "vllm[serve]==0.4.2"        # ← 빌드 없이 1분 내 완료
else
  echo "[vLLM] 이미 설치되어 있습니다."
fi

###############################################################################
# 3. HF 토큰
###############################################################################
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  read -s -p "🔑 HUGGING_FACE_HUB_TOKEN: " TOK; echo
  export HUGGING_FACE_HUB_TOKEN="$TOK"
fi
export HF_HOME="$HF_CACHE"  VLLM_CACHE_ROOT="$VLLM_CACHE"

###############################################################################
# 4. 서버 기동
###############################################################################
echo -e "\n[🚀] vLLM 서버 실행 (포트: $PORT)"
nohup $VENV_PY -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_NAME" \
      --enable_expert_parallel \
      --trust-remote-code \
      --host 0.0.0.0 \
      --port "$PORT"  > "$LOG_FILE" 2>&1 &

PID=$!
printf "[⌛] PID %s – 로딩 중…\n" "$PID"

# 최대 300 초 포트 대기
for _ in {1..150}; do
  sleep 2
  lsof -i :"$PORT" &>/dev/null && {
    echo "✅ vLLM 서버 실행 완료! (PID: $PID)"
    echo "   tail -f $LOG_FILE"
    exit 0
  }
done
echo "❌ 300 초 내에 서버가 열리지 않았습니다. 로그($LOG_FILE) 확인!"
exit 1

