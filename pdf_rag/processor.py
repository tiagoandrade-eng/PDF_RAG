from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import boto3
import fitz

from .analysis_service import AnalysisService
from .config import BATCH_SIZES, TAMANHO_QUEBRA_SIZES, Config
from .database import DatabaseManager
from .embeddings import EmbeddingClient, truncar
from .image_pipeline import ImageContentPipeline, ImageSerializer
from .ocr_service import OCRService
from .text_pipeline import limpar_texto_pagina, serializar_texto_pagina
from .tokens import TokenCounter
from .vision import VisionClient


logger = logging.getLogger(__name__)


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
        self.db = DatabaseManager(self.cfg.rds_db_url, self.cfg.schema)
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
                total_pags,
                self.cfg.schema,
                tabela,
                self.cfg.page_batch_size,
                self.cfg.ocr_workers,
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
                        page_num,
                        s3_key,
                        texto,
                        ja_processados,
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
                "schema": self.cfg.schema,
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
                "=" * 50,
                self.cfg.schema,
                tabela,
                chunks_texto,
                chunks_img,
                chunks_pulados,
                self.counter.resumo(),
                "=" * 50,
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
        self,
        tabela: str,
        texts: list[str],
        metas: list[dict],
        batch_size: int,
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
                    db.insert_records,
                    tabela,
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
