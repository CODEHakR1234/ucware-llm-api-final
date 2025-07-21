#!/bin/bash
# start_services.sh: Redis + Chroma 수동 실행 (포트 변경 반영)

echo "[1] Redis 서버 수동 실행"
redis-server --daemonize yes

sleep 1
echo "[2] Redis 연결 확인"
redis-cli ping || echo "❌ Redis 연결 실패"

echo "[3] Chroma 서버 실행 (포트 9000)"
#nohup chroma run --path ./chroma_db --host 0.0.0.0 --port 9000 --no-auth --no-telemetry > chroma.log 2>&1 &
mkdir -p ./chroma_db
sleep 1

nohup chroma run --path ./chroma_db --host 0.0.0.0 --port 9000 > chroma.log 2>&1 &
# chroma run --path ./chroma_db --host 0.0.0.0 --port 9000

sleep 2
echo "[4] Chroma 상태 확인"
curl -s http://localhost:9000/api/v2/heartbeat || echo "❌ Chroma 연결 실패"

echo "[✔] Redis & Chroma 실행 완료"

