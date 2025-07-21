# app/infra/cache_store.py
from typing import Optional
from app.domain.interfaces import CacheIF
from app.cache.cache_db import get_cache_db  # RedisCacheDB 싱글턴 반환


class CacheStore(CacheIF):
    """CacheIF(Port)를 만족하는 최소 어댑터.

    RedisCacheDB 의 메서드를 그대로 위·아래로 전달만 한다.
    Service 레이어는 CacheIF 타입만 의존하므로,
    나중에 InMemoryCache·MemcachedCache 로도 교체 가능하다.
    """

    def __init__(self):
        self.cache = get_cache_db()   # RedisCacheDB 인스턴스

    # ---- CacheIF 구현 ----
    def get_summary(self, key: str) -> Optional[str]:
        return self.cache.get_pdf(key)

    def set_summary(self, key: str, summary: str) -> None:
        # RedisCacheDB.set_pdf 가 ttl 파라미터를 받아들인다면 전달,
        # 아니라면 기본 TTL 로 저장
        self.cache.set_pdf(key, summary)

    def exists_summary(self, key: str) -> bool:
        return self.cache.exists_pdf(key)
# -------------------------------
# ✅ FastAPI Depends용 provider
# -------------------------------
#_cache_singleton = CacheStore()          # 한 번만 생성

#def get_cache_store() -> CacheIF:
#    """CacheIF 구현체(현재 Redis 기반) 싱글턴을 반환"""
#    return _cache_singleton
