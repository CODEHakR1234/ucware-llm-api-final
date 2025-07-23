
import os
import redis
from functools import lru_cache
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo

class RedisCacheDB:
    def __init__(
        self,
        host: str = os.getenv("REDIS_HOST", "localhost"),
        port: int = int(os.getenv("REDIS_PORT", "6379")),
        db: int = int(os.getenv("REDIS_DB", "0")),
        ttl_days: int = int(os.getenv("REDIS_TTL_DAYS", "7"))
    ):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl_days = ttl_days
        
    def _get_date_key(self, date: datetime = None) -> str:
        """날짜를 기준으로 HSET key 생성"""
        if date is None:
            date = datetime.now(ZoneInfo("Asia/Seoul"))
        return f"pdf:summaries:{date.strftime('%Y-%m-%d')}"
    
    def _get_metadata_key(self, file_id: str) -> str:
        """파일 메타데이터용 key"""
        return f"pdf:metadata:{file_id}"

    def get_pdf(self, fid: str) -> Optional[str]:
        """file_id로 요약본 조회 (모든 날짜에서 검색)"""
        # 1. 먼저 메타데이터에서 저장된 날짜 확인
        metadata_key = self._get_metadata_key(fid)
        metadata = self.r.get(metadata_key)

        if metadata:
            self.r.expire(metadata_key, self.ttl_days * 86400)
            meta = json.loads(metadata)
            date_key = f"pdf:summaries:{meta['date']}"
            summary = self.r.hget(date_key, fid)
            if summary:
                return summary
        
        # 2. 메타데이터가 없으면 최근 7일간의 모든 날짜 검색
        for i in range(self.ttl_days):
            date = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=i)
            date_key = self._get_date_key(date)
            summary = self.r.hget(date_key, fid)
            if summary:
                return summary
        
        return None

    def exists_pdf(self, fid: str) -> bool:
        """해당 file_id 에 대한 요약이 **어디**에든 존재하는지 빠르게 확인."""
        # 1) 메타데이터 키가 있으면 바로 True
        metadata_key = self._get_metadata_key(fid)
        if self.r.exists(metadata_key):
            # 접근 시 TTL 갱신
            self.r.expire(metadata_key, self.ttl_days * 86400)
            return True

        # 2) 최근 ttl_days 동안 날짜별 HSET 조회
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        for i in range(self.ttl_days):
            date_key = self._get_date_key(now - timedelta(days=i))
            if self.r.hexists(date_key, fid):
                return True

        return False

    def set_pdf(self, fid: str, s: str):
        """날짜별 HSET에 요약본 저장"""
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        date_key = self._get_date_key(now)
        
        # 1. HSET에 요약본 저장
        self.r.hset(date_key, fid, s)
        
        # 2. 메타데이터 저장 (조회 성능 향상용)
        metadata = {
            'date': now.strftime('%Y-%m-%d'),
            'timestamp': now.isoformat(),
            'ttl_days': self.ttl_days
        }
        self.r.setex(
            self._get_metadata_key(fid), 
            self.ttl_days * 86400,  # TTL in seconds
            json.dumps(metadata)
        )
        
        # 3. 날짜별 HSET에도 TTL 설정 (8일로 설정해서 여유 확보)
        self.r.expire(date_key, (self.ttl_days + 1) * 86400)

    def set_log(self, file_id: str, url: str, query: str, lang: str, msg: str):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H:%M:%S")

        log_key = f"log:{date_str}"
        log_value = {
            "file_id": file_id,
            "url": url,
            "query": query,
            "lang": lang,
            "time": timestamp,
            "msg": msg
        }
        self.r.hset(log_key, now.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(log_value))

    def delete_pdf(self, fid: str) -> bool:
        metadata_key = self._get_metadata_key(fid)
        metadata = self.r.get(metadata_key)
        deleted = False

        if metadata:
            meta = json.loads(metadata)
            date_key = f"pdf:summaries:{meta['date']}"
            deleted = bool(self.r.hdel(date_key, fid))
            self.r.delete(metadata_key)
        else:
        # 메타데이터가 없으면 최근 날짜 중 찾아서 삭제
            for i in range(self.ttl_days):
                date = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=i)
                date_key = self._get_date_key(date)
                if self.r.hexists(date_key, fid):
                    deleted = bool(self.r.hdel(date_key, fid))
                    break

    # ✅ 삭제 성공했으면 무조건 로그 남기기
        if deleted:
            self._log_cache_deletion(fid)

        return deleted


    def _log_cache_deletion(self, file_id: str):
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        date_str = now.strftime('%Y-%m-%d')
        date_key = f"cache:deleted:{date_str}"
        entry = f"{file_id}|{now.isoformat()}"
        self.r.rpush(date_key, entry)
        print(f"[LOG] Deleted cache entry for {file_id} → {date_key} / {entry}")

    def add_feedback(self, file_id: str, fb_id: str, payload: dict):  # ★
        """
        Key   : feedback:<YYYY-MM-DD>
        Field : <file_id>|<fb_id>|<HH:MM:SS>
        Value : JSON 직렬화된 payload
        TTL   : summaries 정책과 동일 (ttl_days + 1 일)
        """
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        date_key = f"feedback:{now:%Y-%m-%d}"
        field    = f"{file_id}|{fb_id}|{now:%H:%M:%S}"

        self.r.hset(date_key, field, json.dumps(payload))
        self.r.expire(date_key, (self.ttl_days + 1) * 86_400)  # 하루 여유

    def get_feedbacks(self, file_id: str) -> List[dict]:  # ★
        """file_id 에 달린 모든 피드백을 [{…}, …] 형태로 반환"""
        results: List[dict] = []
        for i in range(self.ttl_days + 1):
            date = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=i)
            date_key = f"feedback:{date:%Y-%m-%d}"
            for field, val in self.r.hgetall(date_key).items():
                if field.startswith(f"{file_id}|"):
                    data = json.loads(val)
                    data["id"] = field.split("|")[1]  # fb_id 추출
                    results.append(data)
        return results

@lru_cache(maxsize=1)
def get_cache_db() -> "RedisCacheDB":
    return RedisCacheDB()
