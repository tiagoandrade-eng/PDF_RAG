from __future__ import annotations

import logging
import time
from collections import Counter

import fitz

from .config import EASYOCR_LANGS, MODEL_FULL, MODEL_MINI, Config
from .embeddings import truncar
from .images import extrair_imagem_info, renderizar_pagina
from .ocr_service import OCRService
from .vision import VisionClient, escolher_modelo


logger = logging.getLogger(__name__)


def criar_metadata_imagem(payload: dict, arquivo: str, metodo: str) -> dict:
    return {
        "pagina": payload["page_number"],
        "arquivo": arquivo,
        "chunk_index": payload["chunk_index"],
        "tipo": "imagem",
        "metodo_extracao": metodo,
        "largura_px": payload["width"],
        "altura_px": payload["height"],
        "pagina_renderizada": payload["renderizada"],
        "origem": payload.get("origem", "desconhecida"),
    }


def formatar_texto_imagem(payload: dict, texto: str, metodo: str) -> str:
    label = "SCAN DE PAGINA" if payload["renderizada"] else "IMAGEM TECNICA"
    return truncar(f"[{label} - Pagina {payload['page_number']} | Metodo: {metodo}]\n{texto}")


class ImageSerializer:
    def __init__(self, cfg: Config, xrefs_vistos: set[int]) -> None:
        self.cfg = cfg
        self._xrefs_vistos = xrefs_vistos

    def serializar_paginas(
        self,
        paginas: list[tuple[int, fitz.Page, str]],
        ja_processados: set[tuple[str, str]] | None,
        stats: Counter[str] | None = None,
    ) -> list[dict]:
        payloads: list[dict] = []
        processados = ja_processados or set()

        for page_num, pag, texto_pag in paginas:
            doc = pag.parent
            imagens: list[dict] = []

            for img_info in pag.get_images(full=True):
                xref = img_info[0]
                if xref in self._xrefs_vistos:
                    if stats is not None:
                        stats["duplicate_xrefs_skipped"] += 1
                    continue
                self._xrefs_vistos.add(xref)
                if stats is not None:
                    stats["unique_xrefs_seen"] += 1
                resultado = extrair_imagem_info(
                    doc,
                    xref,
                    min_image_area=self.cfg.min_image_area,
                    min_image_side=self.cfg.min_image_side,
                    max_aspect_ratio=self.cfg.max_aspect_ratio,
                )
                if resultado["ok"]:
                    imagens.append(resultado["payload"])
                elif stats is not None:
                    stats["images_filtered_total"] += 1
                    stats[f"filtered_reason__{resultado['reason']}"] += 1

            if not imagens and not texto_pag:
                render = renderizar_pagina(pag)
                if render:
                    if stats is not None:
                        stats["scan_pages_rendered"] += 1
                    imagens = [
                        {
                            "img_bytes": render.img_bytes,
                            "ext": render.ext,
                            "width": render.width,
                            "height": render.height,
                            "renderizada": True,
                            "origem": render.origem,
                        }
                    ]
                    logger.info("Pag %s: scan detectado, renderizada em %s.", page_num, render.ext)

            for idx, img in enumerate(imagens):
                chunk_index = f"img_{idx}"
                if (str(page_num), chunk_index) in processados:
                    continue
                payloads.append(
                    {
                        "page_number": page_num,
                        "img_idx": idx,
                        "chunk_index": chunk_index,
                        "img_bytes": img["img_bytes"],
                        "ext": img["ext"],
                        "width": img["width"],
                        "height": img["height"],
                        "renderizada": img["renderizada"],
                        "texto_pagina": texto_pag,
                        "langs": EASYOCR_LANGS,
                        "origem": img.get("origem", "desconhecida"),
                    }
                )

        logger.info("Imagens serializadas: %s", len(payloads))
        return payloads


class ImageContentPipeline:
    def __init__(self, cfg: Config, ocr_service: OCRService, vision_client: VisionClient) -> None:
        self.cfg = cfg
        self.ocr_service = ocr_service
        self.vision_client = vision_client

    async def extrair_conteudo(self, payloads: list[dict], arquivo: str) -> tuple[list[str], list[dict]]:
        if not payloads:
            return [], []

        logger.info(
            "Pipeline imagens: %s imagem(ns) | ocr_workers=%s | vision_concurrency=%s",
            len(payloads),
            self.cfg.ocr_workers,
            self.cfg.vision_concurrency,
        )
        ocr_results = await self.ocr_service.executar(payloads)

        aceitos = [payload for payload in ocr_results if payload["chars_uteis"] >= self.cfg.ocr_min_chars]
        vision_needed = [payload for payload in ocr_results if payload["chars_uteis"] < self.cfg.ocr_min_chars]
        n_mini = sum(
            1 for payload in vision_needed if escolher_modelo(payload["width"], payload["height"]) == MODEL_MINI
        )
        logger.info(
            "EasyOCR ok: %s | gpt-4o-mini: %s | gpt-4o: %s",
            len(aceitos),
            n_mini,
            len(vision_needed) - n_mini,
        )

        vision_results: list[dict] = []
        if vision_needed:
            t1 = time.perf_counter()
            vision_results = await self.vision_client.processar_lote(vision_needed)
            logger.info("Vision: %s imagens em %.1fs", len(vision_results), time.perf_counter() - t1)

        textos: list[str] = []
        metas: list[dict] = []

        for payload in aceitos:
            textos.append(formatar_texto_imagem(payload, payload["texto_ocr"], "easyocr"))
            metas.append(criar_metadata_imagem(payload, arquivo, "easyocr"))

        for payload in vision_results:
            texto_vision = payload.get("texto_vision", "").strip()
            if not texto_vision:
                logger.warning("Vision sem resultado: pag %s img_%s. Pulando.", payload["page_number"], payload["img_idx"])
                continue
            modelo = payload.get("modelo_usado", MODEL_FULL)
            textos.append(formatar_texto_imagem(payload, texto_vision, modelo))
            metas.append(criar_metadata_imagem(payload, arquivo, modelo))

        return textos, metas
