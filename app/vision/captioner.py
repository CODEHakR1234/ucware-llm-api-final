# app/vision/captioner.py
"""
Captioner  –  PaliGemma 3B REST 호출
====================================
bytes 이미지 리스트 → 캡션 문자열 리스트 (동일 인덱스 매핑)

환경 변수
---------
CAPTION_ENDPOINT=http://pali:8080/v1      # TGI 서버 /v1 루트
CAPTION_TIMEOUT=30                        # (옵션) 호출 타임아웃
"""

from __future__ import annotations

import asyncio, base64, os
from typing import List
import httpx


class Captioner:
    """멀티모달 배치 캡셔닝 헬퍼."""

    def __init__(self, *, timeout: int | None = None):
        # TGI REST 게이트웨이 기본값 → docker-compose 에서
        self.endpoint = os.getenv("CAPTION_ENDPOINT", "http://pali:8080/v1")
        self._cli = httpx.AsyncClient(
            timeout=int(os.getenv("CAPTION_TIMEOUT", timeout or 30))
        )

    # ─────────────────────────────────────────────────────────
    async def caption(
        self,
        images: List[bytes],
        prompt: str | None = None,
        max_tokens: int = 64,
    ) -> List[str]:

        """
        Args
        ----
        images  : PNG/JPEG bytes 리스트
        prompt  : VLM 프롬프트 (기본 1-2 문장 설명 요청)
        max_tokens : 캡션 최대 토큰

        Returns
        -------
        List[str] : 이미지 순서를 보존한 캡션 문자열 리스트
        """
        if not images:
            return []

        # Captioner 서비스가 없을 경우 기본 캡션 반환
        try:
            prompt = prompt or "Describe this image in 1-2 sentences."

            async def _gen_one(img: bytes) -> str:
                # ① base64 인코딩 → TGI 멀티모달 JSON 스키마
                payload = {
                    "prompt": prompt,
                    "images": [base64.b64encode(img).decode()],
                    "max_new_tokens": max_tokens,
                }
                # ② /generate POST
                r = await self._cli.post(f"{self.endpoint}/generate", json=payload)
                r.raise_for_status()
                # ③ 결과 추출
                return r.json().get("generated_text", "").strip()

            # 여러 장 이미지를 비동기 병렬 처리
            return await asyncio.gather(*(_gen_one(b) for b in images))
        except Exception as e:
            print(f"[Captioner] 캡션 생성 실패, 기본 캡션 사용: {e}", flush=True)
            # 기본 캡션 반환
            return ["이미지" for _ in images]

