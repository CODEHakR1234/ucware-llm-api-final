# app/infra/summarizer.py
"""StuffSummarizer — LCEL Q&A *and* Map‑Reduce summarizer

Implements the revised ``LlmChainIF``::

    class LlmChainIF(Protocol):
        async def execute(self, prompt: str) -> str: ...
        async def summarize(self, docs: List[TextChunk]) -> str: ...

* **execute(prompt)** – takes a *fully‑formatted* prompt string and returns the
  LLM's answer.
* **summarize(docs)** – performs map‑reduce summarization (LangChain's
  ``load_summarize_chain(chain_type="map_reduce")``) on the given text chunks.
"""

from __future__ import annotations

from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from app.utils.llm_factory import get_llm_instance
from app.domain.interfaces import LlmChainIF, TextChunk


class LlmEngine(LlmChainIF):
    """Concrete implementation of :class:`LlmChainIF`."""

    def __init__(self, *, temperature: float = 0.3):
        # Shared LLM instance
        self.llm = get_llm_instance(temperature=temperature)

        # prompt(str) → llm → str  (for *execute*)
        self._qa_chain = (
            RunnablePassthrough()
            | self.llm
            | StrOutputParser()
        )

        # docs(list[str]) → map‑reduce → str (for *summarize*)
        self._summ_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            return_intermediate_steps=False,
        )

    # ------------------------------------------------------------------
    # LlmChainIF implementation
    # ------------------------------------------------------------------
    async def execute(self, prompt: str) -> str:  # noqa: D401
        """LLM call with a fully‑formatted *prompt* string."""
        return (await self._qa_chain.ainvoke(prompt)).strip()

    async def summarize(self, docs: List[TextChunk]) -> str:  # noqa: D401
        """High‑level summary using map‑reduce over *docs*."""
        lc_docs = [Document(page_content=t) for t in docs]
        # ``ainvoke`` returns the final summary string when
        # ``return_intermediate_steps=False``.
        result = await self._summ_chain.ainvoke({"input_documents": lc_docs})

        return str(result["output_text"]).strip()

