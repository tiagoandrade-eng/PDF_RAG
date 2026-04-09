from __future__ import annotations

import logging

from .config import MAX_CHARS


logger = logging.getLogger(__name__)


def limpar_texto_pagina(texto: str) -> str:
    return texto.strip().replace("\x00", "")


def quebrar_texto_em_chunks(texto: str, max_chars: int = MAX_CHARS) -> list[str]:
    return [texto[i : i + max_chars] for i in range(0, len(texto), max_chars)]


def criar_metadata_texto(page_num: int, arquivo: str, chunk_index: int | str) -> dict:
    return {
        "pagina": page_num,
        "arquivo": arquivo,
        "chunk_index": chunk_index,
        "tipo": "texto",
    }


def serializar_texto_pagina(
    page_num: int,
    arquivo: str,
    texto: str,
    ja_processados: set[tuple[str, str]],
) -> tuple[list[str], list[dict], int]:
    if not texto:
        logger.info("Pag %s: sem texto extraivel.", page_num)
        return [], [], 0

    chunks = quebrar_texto_em_chunks(texto)
    logger.info("Pag %s: %s chars -> %s chunk(s)", page_num, len(texto), len(chunks))

    texts: list[str] = []
    metas: list[dict] = []
    pulados = 0

    for idx, chunk in enumerate(chunks):
        if (str(page_num), str(idx)) in ja_processados:
            pulados += 1
            continue
        texts.append(chunk)
        metas.append(criar_metadata_texto(page_num, arquivo, idx))

    return texts, metas, pulados
