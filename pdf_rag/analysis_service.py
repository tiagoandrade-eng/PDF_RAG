from __future__ import annotations

import logging
from collections import Counter

import fitz

from .config import MODEL_FULL, MODEL_MINI, Config
from .image_pipeline import ImageSerializer
from .ocr_service import OCRService
from .text_pipeline import limpar_texto_pagina
from .vision import escolher_modelo


logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, cfg: Config, image_serializer: ImageSerializer, ocr_service: OCRService) -> None:
        self.cfg = cfg
        self.image_serializer = image_serializer
        self.ocr_service = ocr_service

    async def analisar_pdf(self, s3_key: str, pdf_path: str) -> dict:
        doc = fitz.open(pdf_path)
        stats: Counter[str] = Counter()
        payloads: list[dict] = []

        try:
            total_pags = len(doc)
            logger.info(
                "Analise PDF: %s paginas | processar_imagens=%s | ocr_workers=%s",
                total_pags,
                self.cfg.processar_imagens,
                self.cfg.ocr_workers,
            )

            for num in range(total_pags):
                pag = doc[num]
                page_num = num + 1
                texto = limpar_texto_pagina(pag.get_text())
                stats["pages_total"] += 1
                stats["text_chars_total"] += len(texto)
                if texto:
                    stats["pages_with_text"] += 1
                else:
                    stats["pages_without_text"] += 1

                imagens_pagina = pag.get_images(full=True)
                stats["image_occurrences_total"] += len(imagens_pagina)

                if self.cfg.processar_imagens:
                    payloads.extend(self.image_serializer.serializar_paginas([(page_num, pag, texto)], set(), stats))

            stats["images_to_ocr"] = len(payloads)
            ocr_results = await self.ocr_service.executar(payloads)
            aceitos = [payload for payload in ocr_results if payload["chars_uteis"] >= self.cfg.ocr_min_chars]
            vision_needed = [payload for payload in ocr_results if payload["chars_uteis"] < self.cfg.ocr_min_chars]
            mini_count = sum(
                1 for payload in vision_needed if escolher_modelo(payload["width"], payload["height"]) == MODEL_MINI
            )

            resultado = {
                "status": "analysis",
                "arquivo": s3_key,
                "pages_total": stats["pages_total"],
                "pages_with_text": stats["pages_with_text"],
                "pages_without_text": stats["pages_without_text"],
                "avg_text_chars_per_page": round(
                    stats["text_chars_total"] / max(stats["pages_total"], 1),
                    1,
                ),
                "image_occurrences_total": stats["image_occurrences_total"],
                "unique_xrefs_seen": stats["unique_xrefs_seen"],
                "duplicate_xrefs_skipped": stats["duplicate_xrefs_skipped"],
                "images_filtered_total": stats["images_filtered_total"],
                "images_filtered_by_reason": {
                    chave.removeprefix("filtered_reason__"): valor
                    for chave, valor in sorted(stats.items())
                    if chave.startswith("filtered_reason__")
                },
                "scan_pages_rendered": stats["scan_pages_rendered"],
                "images_to_ocr": stats["images_to_ocr"],
                "ocr_accepted": len(aceitos),
                "vision_needed": len(vision_needed),
                "vision_model_split": {
                    MODEL_MINI: mini_count,
                    MODEL_FULL: len(vision_needed) - mini_count,
                },
                "ocr_min_chars": self.cfg.ocr_min_chars,
                "filters": {
                    "min_image_area": self.cfg.min_image_area,
                    "min_image_side": self.cfg.min_image_side,
                    "max_aspect_ratio": self.cfg.max_aspect_ratio,
                },
            }
            logger.info(
                "\n%s\nAnalise concluida\n   Paginas         : %s\n   Ocorrencias img : %s\n"
                "   Xrefs unicos    : %s\n   Duplicadas skip : %s\n   Filtradas       : %s\n"
                "   Para OCR        : %s\n   OCR ok          : %s\n   Iria para Vision: %s\n%s",
                "=" * 50,
                resultado["pages_total"],
                resultado["image_occurrences_total"],
                resultado["unique_xrefs_seen"],
                resultado["duplicate_xrefs_skipped"],
                resultado["images_filtered_total"],
                resultado["images_to_ocr"],
                resultado["ocr_accepted"],
                resultado["vision_needed"],
                "=" * 50,
            )
            return resultado
        finally:
            doc.close()
