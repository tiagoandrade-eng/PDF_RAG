from __future__ import annotations

import asyncio
import json
import sys

from ..models.config import (
    Config,
    carregar_config_execucao,
    configure_logging,
    ler_bool_config,
)
from ..controllers.processor import PDFProcessor


async def processar_documento_async(s3_pdf_key: str, tabela: str, config: Config) -> dict:
    """
    Ponto de entrada assincrono.
    Use este em FastAPI, LangChain agents, ou qualquer contexto async.
    """

    return await PDFProcessor(config).processar(s3_pdf_key, tabela)


async def analisar_documento_async(s3_pdf_key: str, config: Config) -> dict:
    """
    Analise previa sem uso de OpenAI.
    """

    return await PDFProcessor(config).analisar(s3_pdf_key)


def processar_documento_api(
    s3_pdf_key: str,
    tabela: str,
    rds_db_url: str,
    openai_api_key: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str,
    aws_bucket_name: str,
    schema: str = "public",
    processar_imagens: bool = True,
    page_batch_size: int = 100,
    ocr_workers: int = 4,
    vision_concurrency: int = 6,
    min_image_area: int = 10_000,
    min_image_side: int = 80,
    max_aspect_ratio: float = 8.0,
    ocr_min_chars: int = 60,
) -> dict:
    """
    Ponto de entrada sincrono.
    Use este em scripts standalone, workers Celery e chamadas diretas.
    """

    configure_logging()
    cfg = Config(
        openai_api_key=openai_api_key,
        rds_db_url=rds_db_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
        aws_bucket_name=aws_bucket_name,
        db_schema=schema,
        processar_imagens=processar_imagens,
        page_batch_size=page_batch_size,
        ocr_workers=ocr_workers,
        vision_concurrency=vision_concurrency,
        min_image_area=min_image_area,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
        ocr_min_chars=ocr_min_chars,
    )
    return asyncio.run(processar_documento_async(s3_pdf_key, tabela, cfg))


def analisar_documento_api(
    s3_pdf_key: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str,
    aws_bucket_name: str,
    schema: str = "public",
    processar_imagens: bool = True,
    page_batch_size: int = 100,
    ocr_workers: int = 4,
    vision_concurrency: int = 6,
    min_image_area: int = 10_000,
    min_image_side: int = 80,
    max_aspect_ratio: float = 8.0,
    ocr_min_chars: int = 60,
) -> dict:
    configure_logging()
    cfg = Config(
        openai_api_key="",
        rds_db_url="",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
        aws_bucket_name=aws_bucket_name,
        db_schema=schema,
        processar_imagens=processar_imagens,
        page_batch_size=page_batch_size,
        ocr_workers=ocr_workers,
        vision_concurrency=vision_concurrency,
        min_image_area=min_image_area,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
        ocr_min_chars=ocr_min_chars,
        modo_analise=True,
    )
    return asyncio.run(analisar_documento_async(s3_pdf_key, cfg))


def main() -> int:
    configure_logging()
    try:
        runtime = carregar_config_execucao()
        common_kwargs = {
            "s3_pdf_key": runtime.get("S3_PDF_KEY", "suporte/seu_arquivo.pdf"),
            "schema": runtime.get("DB_SCHEMA", "public"),
            "processar_imagens": True,
            "page_batch_size": 100,
            "ocr_workers": 4,
            "vision_concurrency": 6,
            "ocr_min_chars": 60,
            "min_image_area": 10_000,
            "min_image_side": 80,
            "max_aspect_ratio": 8.0,
            "aws_access_key_id": runtime["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": runtime["AWS_SECRET_ACCESS_KEY"],
            "aws_region": runtime.get("AWS_REGION", "us-east-1"),
            "aws_bucket_name": runtime["AWS_BUCKET_NAME"],
        }
        modo_analise = ler_bool_config(runtime.get("MODO_ANALISE"), False)

        if modo_analise:
            resultado = analisar_documento_api(
                **common_kwargs,
            )
        else:
            resultado = processar_documento_api(
                tabela=runtime.get("TABELA_RAG", "minha_tabela_rag"),
                rds_db_url=runtime["RDS_DB_URL"],
                openai_api_key=runtime["OPENAI_API_KEY"],
                **common_kwargs,
            )
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
        return 0
    except KeyError as exc:
        print(
            f"Configuracao obrigatoria ausente: {exc.args[0]}. "
            "Preencha .env e app_settings.env antes de rodar.",
            file=sys.stderr,
        )
        return 1
