from __future__ import annotations
import re
from typing import List, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.domain.page_element import PageElement
from app.domain.page_chunk   import PageChunk          # ★ NEW

# ──────────────── 하이퍼 파라미터 ────────────────
_MAX, _OVF, _OVL = 1_500, 1_800, 200
sent_split = RecursiveCharacterTextSplitter(
    chunk_size=_MAX,
    chunk_overlap=_OVL,
    separators=[". ", "!\n", "?\n"],
)

_HDR_H1 = re.compile(r"^((\d+[\.-])+ )?[IVXLCDM0-9]+\.\s|^[A-Z][A-Z\s\-]{3,}$")
_HDR_H2 = re.compile(r"^\d+\.\d+\.?\s+.+")
_HDR_H3 = re.compile(r"^\d+\.\d+\.\d+\.?\s+.+")
_BULLET = re.compile(r"^(\s*[\u2022\u2023\u25CF\-\*])|^\s*\d+\.\s+")
_PAR_BR = re.compile(r"\n{2,}")

# ─────────────────────────────────────────────────
class SemanticChunker:
    """
    PageElement 리스트 → semantic chunk 리스트로 변환.

    Parameters
    ----------
    return_pagechunk : True 이면 PageChunk 객체,
                       False 이면 plain str 을 돌려준다.
    """

    def group(
        self,
        els: List[PageElement],
        *,
        return_pagechunk: bool = False
    ) -> List[Union[str, PageChunk]]:
        blocks, buf, figs = [], [], []

        def flush(page_no: int):
            """버퍼 내용을 하나의 청크로 밀어 넣는다."""
            if not buf:
                return

            joined = " ".join(buf).strip()
            texts  = (
                sent_split.split_text(joined)
                if len(joined) > _OVF
                else [joined]
            )

            if return_pagechunk:
                blocks.extend(
                    PageChunk(page=page_no, text=t, figs=list(figs))
                    for t in texts
                )
            else:
                blocks.extend(texts)

            buf.clear()
            figs.clear()

        last_page = -1

        for el in els:
            # 페이지가 바뀌면 flush
            if el.page_no != last_page:
                flush(last_page)
                last_page = el.page_no

            if el.kind == "text":
                for p in _PAR_BR.split(el.content):
                    p = p.strip()
                    if not p:
                        continue
                    if _HDR_H1.match(p):
                        flush(el.page_no)
                        blocks.append(
                            PageChunk(el.page_no, f"# {p}") if return_pagechunk else f"# {p}"
                        )
                    elif _HDR_H2.match(p):
                        flush(el.page_no)
                        buf.append(f"## {p}")
                    elif _HDR_H3.match(p):
                        flush(el.page_no)
                        buf.append(f"### {p}")
                    elif _BULLET.match(p):
                        buf.append(p)
                    else:
                        buf.append(p)
            else:  # figure / table / graph
                img_md = f"![{el.caption or ''}]({el.content})"
                buf.append(img_md)
                figs.append(el.content)

            if sum(len(x) for x in buf) > _MAX + _OVL:
                flush(el.page_no)

        flush(last_page)
        return blocks

