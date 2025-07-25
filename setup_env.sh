#!/bin/bash
set -e

echo "[1] 시스템 패키지 설치"
sudo apt update
sudo apt install -y redis-server libgl1 libglib2.0-0

echo "[2] Python 가상환경 생성 및 활성화"
python3 -m venv .venv
source .venv/bin/activate

echo "[3] pip 패키지 설치 (공통)"
pip install --upgrade pip
pip install -r requirements.txt   # FastAPI/LangChain 등

#───────────────────────── PaddleOCR 설치 ────────────────────────#
#   cu11.8 & cu12.x 모두 paddlepaddle-gpu==2.6.2.post118 휠 사용
#   (CUDA 11.8 toolchain이 12.x 드라이버와 호환)
PADDLE_WHEEL="paddlepaddle-gpu==2.6.2.post118"
PADDLE_EXTRA_URL="-f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html"

echo "[2] PaddlePaddle GPU 휠 설치 (${PADDLE_WHEEL})"
pip install "${PADDLE_WHEEL}" paddleocr rapidfuzz pillow ${PADDLE_EXTRA_URL}

# ──────────────── LLM / Embedding Provider 선택 ────────────────
echo ""
echo "🤖 사용할 LLM/Embedding Provider를 선택하세요:"
echo "1. openai"
echo "2. hf (HuggingFace)"
read -p "선택 [1/2]: " PROVIDER_CHOICE

echo -n "🔑 Tavily API Key를 입력하세요: "
read -r TAVILY_API_KEY

echo -n "🎮 사용할 GPU 번호를 입력하세요 (여러 장이면 콤마, 기본 0): "
read -r GPU_NUMBER
GPU_NUMBER=${GPU_NUMBER:-0}

if [ "$PROVIDER_CHOICE" == "2" ]; then
    LLM_PROVIDER="hf"
    EMBEDDING_MODEL_NAME="BAAI/bge-m3"
    LLM_MODEL_NAME="Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    OPENAI_API_KEY=""
else
    LLM_PROVIDER="openai"
    EMBEDDING_MODEL_NAME="text-embedding-ada-002"
    LLM_MODEL_NAME="gpt-3.5-turbo"
    echo -n "🔑 OpenAI API Key를 입력하세요: "
    read -r OPENAI_API_KEY
fi

# ──────────────── .env 생성 ────────────────
echo "[4] .env 파일 생성"
cat > .env <<EOF
CHROMA_HOST=localhost
CHROMA_PORT=9000
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_TTL=604800
LLM_PROVIDER=$LLM_PROVIDER
EMBEDDING_MODEL_NAME=$EMBEDDING_MODEL_NAME
LLM_MODEL_NAME=$LLM_MODEL_NAME
OPENAI_API_KEY="$OPENAI_API_KEY"
TAVILY_API_KEY="$TAVILY_API_KEY"
CUDA_VISIBLE_DEVICES=$GPU_NUMBER
# PaddleOCR 전용
PADDLE_GPU_MEM=8000
EOF

echo "[✔] 환경 구성이 완료되었습니다."
echo "가상환경 활성화:  source .venv/bin/activate"

