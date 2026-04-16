from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections import Counter

import boto3
import fitz
from openai import AsyncOpenAI

from ..models.config import Config, TokenCounter, truncar
from ..models.constants import (
    BATCH_SIZES,
    MAX_CHARS,
    MODEL_FULL,
    MODEL_MINI,
    TAMANHO_QUEBRA_SIZES,
)
from ..models.database import DatabaseManager
from .images import (
    ImageContentPipeline,
    ImageSerializer,
    OCRService,
    VisionClient,
    escolher_modelo,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
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
                        attempt + 1, exc, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise


# ---------------------------------------------------------------------------
# Text pipeline
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Analysis service (modo analise)
# ---------------------------------------------------------------------------
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
                total_pags, self.cfg.processar_imagens, self.cfg.ocr_workers,
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
                    stats["text_chars_total"] / max(stats["pages_total"], 1), 1,
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
                resultado["pages_total"], resultado["image_occurrences_total"],
                resultado["unique_xrefs_seen"], resultado["duplicate_xrefs_skipped"],
                resultado["images_filtered_total"], resultado["images_to_ocr"],
                resultado["ocr_accepted"], resultado["vision_needed"],
                "=" * 50,
            )
            return resultado
        finally:
            doc.close()


# ---------------------------------------------------------------------------
# PDF Processor (orquestrador principal)
# ---------------------------------------------------------------------------
class PDFProcessor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._xrefs_vistos: set[int] = set()
        self.counter = TokenCounter()
        self.db: DatabaseManager | None = None
        self.embedder: EmbeddingClient | None = None
        self.vision: VisionClient | None = None
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=cfg.aws_access_key_id,
            aws_secret_access_key=cfg.aws_secret_access_key,
            region_name=cfg.aws_region,
        )

    async def processar(self, s3_key: str, tabela: str) -> dict:
        self._inicializar_clientes_processamento()
        db, _, vision = self._require_processing_clients()
        tabela = db.garantir_tabela(tabela)
        self._resetar_estado_documento()

        tmp_path = await self._baixar_pdf_temporario(s3_key)
        try:
            image_serializer = ImageSerializer(self.cfg, self._xrefs_vistos)
            image_pipeline = ImageContentPipeline(self.cfg, OCRService(self.cfg.ocr_workers), vision)
            return await self._pipeline(s3_key, tabela, tmp_path, image_serializer, image_pipeline)
        finally:
            self._remover_temp(tmp_path)

    async def analisar(self, s3_key: str) -> dict:
        self._resetar_estado_documento()

        tmp_path = await self._baixar_pdf_temporario(s3_key, analise=True)
        try:
            image_serializer = ImageSerializer(self.cfg, self._xrefs_vistos)
            analysis_service = AnalysisService(self.cfg, image_serializer, OCRService(self.cfg.ocr_workers))
            return await analysis_service.analisar_pdf(s3_key, tmp_path)
        finally:
            self._remover_temp(tmp_path)

    def _inicializar_clientes_processamento(self) -> None:
        self.db = DatabaseManager(self.cfg.rds_db_url, self.cfg.db_schema)
        self.embedder = EmbeddingClient(self.cfg.openai_api_key, self.counter)
        self.vision = VisionClient(self.cfg.openai_api_key, self.counter, self.cfg.vision_concurrency)

    def _require_processing_clients(self) -> tuple[DatabaseManager, EmbeddingClient, VisionClient]:
        if self.db is None or self.embedder is None or self.vision is None:
            raise RuntimeError("Clientes de processamento nao foram inicializados.")
        return self.db, self.embedder, self.vision

    def _resetar_estado_documento(self) -> None:
        self._xrefs_vistos = set()

    async def _baixar_pdf_temporario(self, s3_key: str, analise: bool = False) -> str:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)
        sufixo_log = " para analise" if analise else ""
        logger.info("Baixando %s/%s%s", self.cfg.aws_bucket_name, s3_key, sufixo_log)
        await asyncio.to_thread(self.s3.download_file, self.cfg.aws_bucket_name, s3_key, tmp_path)
        return tmp_path

    def _remover_temp(self, tmp_path: str) -> None:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("Temp removido: %s", tmp_path)

    async def _pipeline(
        self,
        s3_key: str,
        tabela: str,
        pdf_path: str,
        image_serializer: ImageSerializer,
        image_pipeline: ImageContentPipeline,
    ) -> dict:
        db, _, _ = self._require_processing_clients()
        ja_processados = db.buscar_processados(tabela, s3_key)
        doc = fitz.open(pdf_path)

        try:
            total_pags = len(doc)
            logger.info(
                "PDF: %s paginas | schema='%s' | tabela='%s' | batch=%s | ocr_workers=%s",
                total_pags, self.cfg.db_schema, tabela, self.cfg.page_batch_size, self.cfg.ocr_workers,
            )

            chunks_texto = 0
            chunks_img = 0
            chunks_pulados = 0
            batch_texts: list[str] = []
            batch_metas: list[dict] = []

            for start in range(0, total_pags, self.cfg.page_batch_size):
                end = min(start + self.cfg.page_batch_size, total_pags)
                logger.info("== Batch paginas %s-%s ==", start + 1, end)

                paginas_img: list[tuple[int, fitz.Page, str]] = []

                for num in range(start, end):
                    pag = doc[num]
                    page_num = num + 1
                    texto = limpar_texto_pagina(pag.get_text())

                    textos_pagina, metas_pagina, pulados_pagina = serializar_texto_pagina(
                        page_num, s3_key, texto, ja_processados,
                    )
                    chunks_pulados += pulados_pagina

                    for chunk, meta in zip(textos_pagina, metas_pagina):
                        batch_texts.append(chunk)
                        batch_metas.append(meta)
                        if len(batch_texts) >= BATCH_SIZES[0]:
                            chunks_texto += await self._flush(tabela, batch_texts, batch_metas)
                            batch_texts, batch_metas = [], []

                    if self.cfg.processar_imagens:
                        paginas_img.append((page_num, pag, texto))

                if self.cfg.processar_imagens and paginas_img:
                    payloads = image_serializer.serializar_paginas(paginas_img, ja_processados)
                    textos_img, metas_img = await image_pipeline.extrair_conteudo(payloads, s3_key)
                    if textos_img:
                        chunks_img += await self._flush(tabela, textos_img, metas_img)

            if batch_texts:
                chunks_texto += await self._flush(tabela, batch_texts, batch_metas)

            tokens_resumo = {
                modelo: {"input_tokens": contagem["input"], "output_tokens": contagem["output"]}
                for modelo, contagem in self.counter.tokens.items()
            }
            resultado = {
                "status": "ok",
                "schema": self.cfg.db_schema,
                "tabela": tabela,
                "chunks_texto": chunks_texto,
                "chunks_imagem": chunks_img,
                "chunks_pulados": chunks_pulados,
                "openai_tokens": tokens_resumo,
                "openai_custo_usd": round(self.counter.custo_usd(), 6),
            }
            logger.info(
                "\n%s\nConcluido!\n   Schema : %s\n   Tabela : %s\n   Texto  : %s chunks inseridos\n"
                "   Imagens: %s chunks inseridos\n   Pulados: %s (ja existiam)\n%s\n%s",
                "=" * 50, self.cfg.db_schema, tabela, chunks_texto, chunks_img,
                chunks_pulados, self.counter.resumo(), "=" * 50,
            )
            return resultado
        finally:
            doc.close()

    async def _flush(self, tabela: str, texts: list[str], metas: list[dict]) -> int:
        if not texts:
            return 0
        for batch_size in BATCH_SIZES:
            try:
                return await self._flush_em_blocos(tabela, texts, metas, batch_size)
            except Exception as exc:
                logger.warning("Flush falhou com batch_size=%s: %s", batch_size, exc)
        return await self._flush_com_quebra(tabela, texts, metas)

    async def _flush_em_blocos(
        self, tabela: str, texts: list[str], metas: list[dict], batch_size: int,
    ) -> int:
        db, embedder, _ = self._require_processing_clients()
        total = 0
        for index in range(0, len(texts), batch_size):
            bloco_t = [truncar(texto) for texto in texts[index : index + batch_size]]
            bloco_m = metas[index : index + batch_size]
            vetores = await embedder.embed(bloco_t)
            records = [
                {"text": bloco_t[i], "metadata": bloco_m[i], "embedding": vetores[i]}
                for i in range(len(bloco_t))
            ]
            total += await asyncio.to_thread(db.insert_records, tabela, records)
        return total

    async def _flush_com_quebra(self, tabela: str, texts: list[str], metas: list[dict]) -> int:
        db, embedder, _ = self._require_processing_clients()
        total = 0
        for texto, meta in zip(texts, metas):
            texto = truncar(texto)
            try:
                vetores = await embedder.embed([texto])
                total += await asyncio.to_thread(
                    db.insert_records, tabela,
                    [{"text": texto, "metadata": meta, "embedding": vetores[0]}],
                )
            except Exception as exc:
                logger.warning("Embedding falhou pag %s: %s. Quebrando...", meta["pagina"], exc)
                for tam in TAMANHO_QUEBRA_SIZES:
                    subs = [texto[i : i + tam] for i in range(0, len(texto), tam)]
                    salvos: list[dict] = []
                    falhou = False
                    for sub_index, sub in enumerate(subs):
                        try:
                            vetor = await embedder.embed([sub])
                            salvos.append(
                                {
                                    "text": sub,
                                    "metadata": {
                                        **meta,
                                        "chunk_index": f"{meta['chunk_index']}_sub{sub_index}",
                                        "quebrado": True,
                                    },
                                    "embedding": vetor[0],
                                }
                            )
                        except Exception:
                            falhou = True
                            break
                    if not falhou:
                        for record in salvos:
                            total += await asyncio.to_thread(db.insert_records, tabela, [record])
                        break
                else:
                    logger.error("Falha total pag %s. Pulando.", meta["pagina"])
        return total
