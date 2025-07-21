
# 📄 PDF Summary API 사용 가이드

---

## ✅ 가상환경 활성화 및 서버 프로세스 확인
```bash
source .venv/bin/activate
ps aux | grep uvicorn
```

---

## 📌 요약 요청 (Summary 생성)

### 0. Attention is All you Need
```bash
curl -X POST http://localhost:8000/summary \
     -H "Content-Type: application/json" \
     -d '{
           "file_id": "paper-01",
           "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"
         }'
```

### 1. Transformer (NLP) - "Attention is All You Need"
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{"file_id": "nlp-transformer", "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"}'
```

### 2. AlphaGo (Reinforcement Learning + MCTS + Deep Learning)
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{"file_id": "nlp-transformer", "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"}'
```

### 3. GCN (Graph Neural Networks) - Semi-supervised classification with GCN
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{"file_id": "gcn-graph", "pdf_url": "https://arxiv.org/pdf/1609.02907.pdf"}'
```

### 4. Stable Diffusion (Text-to-Image Generation)
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{"file_id": "gen-stablediff", "pdf_url": "https://arxiv.org/pdf/2112.10752.pdf"}'
```

### 5. Segment Anything Model (Computer Vision, Foundation Model)
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{"file_id": "cv-sam", "pdf_url": "https://arxiv.org/pdf/2304.02643.pdf"}'
```

### 6. DINOv2 (Self-supervised Learning for Vision)
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{"file_id": "cv-dinov2", "pdf_url": "https://arxiv.org/pdf/2304.07193.pdf"}'
```

### 7. Semiconductor
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "semiconductor-memory",
    "pdf_url": "https://arxiv.org/pdf/1905.06962.pdf"
  }'
```

### 8. Quantum Computing
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "quantum-computing-review",
    "pdf_url": "https://arxiv.org/pdf/1903.04500.pdf"
  }'
```

---

## 🧠 벡터 (VectorDB) 관리 기능

### A. Vector 조회
1. **벡터 통계 확인**  
   저장된 file_id 리스트 확인
   ```bash
   curl -X GET http://localhost:8000/vector/statistics
   ```

2. **벡터 존재 확인**  
   저장된 file_id 리스트 확인
   ```bash
   curl -X GET http://localhost:8000/vector/check/quantum-computing-review
   ```

3. **날짜별 벡터 조회**  
   날짜별 저장된 벡터 조회 가능
   ```bash
   curl -X GET "http://localhost:8000/vector/by-date?date=2025-07-14"
   ```

---

### B. Vector 삭제
1. **벡터 수동 정리** (캐시에 없는 벡터 삭제)
   ```bash
   curl -X DELETE http://localhost:8000/vector/cleanup-unused
   ```

2. **벡터 특정 파일 수동 삭제**
   ```bash
   curl -X DELETE http://localhost:8000/vector/delete/semiconductor-memory
   ```

3. **벡터 전체 데이터 삭제**
   ```bash
   curl -X DELETE http://localhost:8000/vector/all
   ```

---

### C. Vector 로그 관리
1. **벡터 삭제 로그 날짜별 조회**
   ```bash
   curl -X GET "http://localhost:8000/vector/cleanup-log?date=2025-07-14"
   ```

2. **벡터 삭제 로그 삭제**
   ```bash
   curl -X DELETE "http://localhost:8000/vector/cleanup-log?date=2025-07-14"
   ```

---

## 🧊 캐시 (Redis) 관리 기능

### A. Cache 조회
1. **캐시 통계 확인**  
   요약본 캐시에 저장된 file_id 개수 + 메모리 사용량 확인
   ```bash
   curl -X GET http://localhost:8000/cache/statistics
   ```

2. **캐시 존재 확인**
   ```bash
   curl -X GET http://localhost:8000/cache/check/quantum-computing-review
   ```

3. **날짜별 캐시 조회**
   ```bash
   curl -X GET http://localhost:8000/cache/summaries/2025-07-14
   ```

---

### B. Cache 삭제
1. **TTL 지난 캐시 수동 정리**
   ```bash
   curl -X DELETE http://localhost:8000/cache/cleanup
   ```

2. **캐시 특정 파일 수동 삭제**
   ```bash
   curl -X DELETE http://localhost:8000/cache/summary/quantum-computing-review
   ```

3. **캐시 전체 데이터 삭제**
   ```bash
   curl -X DELETE http://localhost:8000/cache/all
   ```

---

### C. Cache 로그 관리
1. **캐시 삭제 로그 날짜별 조회**
   ```bash
   curl "http://localhost:8000/cache/deletion-log?date=2025-07-14"
   ```

2. **캐시 메타데이터 확인** (⚠️ *현재 Internal Server Error*)
   ```bash
   curl -X GET http://localhost:8000/cache/metadata/quantum-computing-review
   ```

3. **캐시 삭제 로그 삭제**
   ```bash
   curl -X DELETE "http://localhost:8000/cache/deletion-log?date=2025-07-14"
   ```

---

## 🧹 캐시 & 벡터 전체 삭제
1. **Vector & 캐시 전체 데이터 삭제** (메타데이터 포함)
   ```bash
   curl -X DELETE http://localhost:8000/system/all
   ```
