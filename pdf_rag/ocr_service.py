from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor

from .config import EASYOCR_LANGS
from .images import _init_easyocr_worker, easyocr_worker


logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self, workers: int) -> None:
        self.workers = workers

    async def executar(self, payloads: list[dict]) -> list[dict]:
        if not payloads:
            return []

        total = len(payloads)
        logger.info("EasyOCR iniciando: %s imagens | workers=%s", total, self.workers)

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        langs = payloads[0].get("langs", EASYOCR_LANGS)
        resultados: list[dict] = []
        concluidos = 0

        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_easyocr_worker,
            initargs=(langs,),
        ) as pool:
            futures = [loop.run_in_executor(pool, easyocr_worker, payload) for payload in payloads]
            for fut in asyncio.as_completed(futures):
                resultado = await fut
                resultados.append(resultado)
                concluidos += 1
                if concluidos % 10 == 0 or concluidos == total:
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "EasyOCR: %s/%s imagens (%.0f%%) | %.1fs decorridos",
                        concluidos,
                        total,
                        100 * concluidos / total,
                        elapsed,
                    )

        logger.info("EasyOCR concluido: %s imagens em %.1fs", total, time.perf_counter() - t0)
        return resultados
