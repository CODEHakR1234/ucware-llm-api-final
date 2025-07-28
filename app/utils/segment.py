"""
segment.py
----------
물리적 페이지 순서*를 유지*하면서 **같은 주제**로 보이는 연속 블록을 묶는다.

입력 : PageChunk 리스트(페이지 순)  
출력 : (페이지번호 리스트, PageChunk 리스트) 튜플들의 리스트
"""
from typing import List, Tuple, Callable
import numpy as np
from scipy.spatial.distance import cosine
from app.domain.page_chunk import PageChunk


def segment_in_order(
    chunks: List[PageChunk],
    embed_fn: Callable[[str], np.ndarray],
    *,
    sim_threshold: float = 0.78,
    max_gap_pages: int = 1,
) -> List[Tuple[List[int], List[PageChunk]]]:
    """
    페이지 흐름을 지키면서 *주제 연속성*을 기준으로 세그먼트를 나눈다.
    """
    if not chunks:
        return []

    vecs = [embed_fn(c.text) for c in chunks]

    segments: List[Tuple[List[int], List[PageChunk], np.ndarray]] = []
    cur_pages, cur_chunks = [chunks[0].page], [chunks[0]]
    centroid = vecs[0]  # 현재 세그먼트 중심벡터

    for ck, v in zip(chunks[1:], vecs[1:]):
        gap = ck.page - cur_pages[-1]           # 페이지 끊김 정도
        sim = 1 - cosine(centroid, v)           # 내용 유사도

        if sim >= sim_threshold and gap <= max_gap_pages:
            # 같은 세그먼트에 편입
            cur_pages.append(ck.page)
            cur_chunks.append(ck)
            centroid = np.mean(np.vstack([centroid, v]), axis=0)
        else:
            # 새 세그먼트 시작
            segments.append((cur_pages, cur_chunks, centroid))
            cur_pages, cur_chunks, centroid = [ck.page], [ck], v

    segments.append((cur_pages, cur_chunks, centroid))  # 마지막 flush
    return [(pages, cks) for pages, cks, _ in segments]

