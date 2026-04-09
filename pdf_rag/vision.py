from __future__ import annotations

import asyncio
import base64
import logging

import httpx

from .config import FORMATOS_SUPORTADOS, IMAGE_AREA_MINI_THRESHOLD, MODEL_FULL, MODEL_MINI
from .tokens import TokenCounter


logger = logging.getLogger(__name__)


def escolher_modelo(width: int, height: int) -> str:
    return MODEL_FULL if (width * height) >= IMAGE_AREA_MINI_THRESHOLD else MODEL_MINI


class VisionClient:
    _MIME = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    _SYSTEM = (
        "Voce e um especialista em manuais tecnicos de manutencao e servico. "
        "Analise a imagem e forneca descricao tecnica detalhada em portugues. "
        "Identifique componentes, pecas, numeracao de referencia, setas, medidas, "
        "especificacoes e sequencia de montagem/desmontagem, se houver. "
        "Transcreva integralmente qualquer texto visivel na imagem. "
        "Seja especifico: esta descricao sera usada para busca semantica."
    )

    def __init__(self, api_key: str, counter: TokenCounter, concurrency: int) -> None:
        self._api_key = api_key
        self._counter = counter
        self._sem = asyncio.Semaphore(concurrency)

    async def descrever(
        self, session: httpx.AsyncClient, payload: dict, max_retries: int = 3
    ) -> dict:
        ext = payload.get("ext", "").lower()
        if ext not in FORMATOS_SUPORTADOS:
            return {**payload, "texto_vision": "", "modelo_usado": MODEL_FULL}

        modelo = escolher_modelo(payload.get("width", 0), payload.get("height", 0))
        mime = self._MIME[ext]
        b64 = base64.b64encode(payload["img_bytes"]).decode()
        contexto = payload.get("texto_pagina", "")[:400]
        prompt = "Descreva esta imagem tecnica detalhadamente."
        if contexto:
            prompt += f"\n\nContexto da pagina:\n{contexto}"

        body = {
            "model": modelo,
            "max_tokens": 1_000,
            "messages": [
                {"role": "system", "content": self._SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }

        async with self._sem:
            for attempt in range(max_retries):
                try:
                    resp = await session.post(
                        "https://api.openai.com/v1/chat/completions",
                        json=body,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        timeout=60.0,
                    )
                    if resp.status_code != 200:
                        logger.error(
                            "Vision HTTP %s pag %s: %s",
                            resp.status_code,
                            payload["page_number"],
                            resp.text[:300],
                        )
                    resp.raise_for_status()
                    resposta_json = resp.json()
                    texto = resposta_json["choices"][0]["message"]["content"].strip()
                    uso = resposta_json.get("usage", {})
                    self._counter.add(
                        modelo,
                        uso.get("prompt_tokens", 0),
                        uso.get("completion_tokens", 0),
                    )
                    logger.info(
                        "Vision pag %s img_%s [%s]: %s chars",
                        payload["page_number"],
                        payload["img_idx"],
                        modelo,
                        len(texto),
                    )
                    return {**payload, "texto_vision": texto, "modelo_usado": modelo}
                except httpx.HTTPStatusError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        return {**payload, "texto_vision": "", "modelo_usado": modelo}
                except Exception as exc:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        logger.error(
                            "Vision falhou pag %s img_%s: %s",
                            payload["page_number"],
                            payload["img_idx"],
                            exc,
                        )
                        return {**payload, "texto_vision": "", "modelo_usado": modelo}

    async def processar_lote(self, payloads: list[dict]) -> list[dict]:
        async with httpx.AsyncClient() as session:
            return list(await asyncio.gather(*[self.descrever(session, payload) for payload in payloads]))
