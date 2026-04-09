from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from .config import MAX_CHARS
from .tokens import TokenCounter


logger = logging.getLogger(__name__)


def truncar(texto: str) -> str:
    if len(texto) > MAX_CHARS:
        logger.warning("Texto truncado: %s -> %s chars.", len(texto), MAX_CHARS)
        return texto[:MAX_CHARS]
    return texto


class EmbeddingClient:
    MODEL = "text-embedding-3-large"

    def __init__(self, api_key: str, counter: TokenCounter) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._counter = counter

    async def embed(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        for attempt in range(max_retries):
            try:
                resp = await self._client.embeddings.create(input=texts, model=self.MODEL)
                if resp.usage:
                    self._counter.add(self.MODEL, resp.usage.total_tokens)
                return [item.embedding for item in resp.data]
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    logger.warning(
                        "Embedding erro (tentativa %s): %s. Aguardando %ss...",
                        attempt + 1,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
