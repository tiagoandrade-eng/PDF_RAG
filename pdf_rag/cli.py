from __future__ import annotations

import asyncio
import json
import sys

from .config import (
    DEFAULT_SCHEMA,
    MAX_ASPECT_RATIO,
    MIN_IMAGE_AREA,
    MIN_IMAGE_SIDE,
    OCR_MIN_CHARS,
    OCR_WORKERS_DEFAULT,
    PAGE_BATCH_SIZE,
    VISION_CONCURRENCY,
    Config,
    carregar_config_execucao,
    configure_logging,
    ler_bool_config,
    ler_float_config,
    ler_int_config,
)
from .processor import PDFProcessor


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
    schema: str = DEFAULT_SCHEMA,
    processar_imagens: bool = True,
    page_batch_size: int = PAGE_BATCH_SIZE,
    ocr_workers: int = OCR_WORKERS_DEFAULT,
    vision_concurrency: int = VISION_CONCURRENCY,
    min_image_area: int = MIN_IMAGE_AREA,
    min_image_side: int = MIN_IMAGE_SIDE,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
    ocr_min_chars: int = OCR_MIN_CHARS,
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
        schema=schema,
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
    schema: str = DEFAULT_SCHEMA,
    processar_imagens: bool = True,
    page_batch_size: int = PAGE_BATCH_SIZE,
    ocr_workers: int = OCR_WORKERS_DEFAULT,
    vision_concurrency: int = VISION_CONCURRENCY,
    min_image_area: int = MIN_IMAGE_AREA,
    min_image_side: int = MIN_IMAGE_SIDE,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
    ocr_min_chars: int = OCR_MIN_CHARS,
) -> dict:
    configure_logging()
    cfg = Config(
        openai_api_key="",
        rds_db_url="",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
        aws_bucket_name=aws_bucket_name,
        schema=schema,
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
        modo_analise = ler_bool_config(runtime.get("MODO_ANALISE"), False)
        common_kwargs = {
            "s3_pdf_key": runtime.get("S3_PDF_KEY", "suporte/seu_arquivo.pdf"),
            "schema": runtime.get("DB_SCHEMA", DEFAULT_SCHEMA),
            "processar_imagens": ler_bool_config(runtime.get("PROCESSAR_IMAGENS"), True),
            "page_batch_size": ler_int_config(runtime.get("PAGE_BATCH_SIZE"), PAGE_BATCH_SIZE),
            "ocr_workers": ler_int_config(runtime.get("OCR_WORKERS"), OCR_WORKERS_DEFAULT),
            "vision_concurrency": ler_int_config(runtime.get("VISION_CONCURRENCY"), VISION_CONCURRENCY),
            "min_image_area": ler_int_config(runtime.get("MIN_IMAGE_AREA"), MIN_IMAGE_AREA),
            "min_image_side": ler_int_config(runtime.get("MIN_IMAGE_SIDE"), MIN_IMAGE_SIDE),
            "max_aspect_ratio": ler_float_config(runtime.get("MAX_ASPECT_RATIO"), MAX_ASPECT_RATIO),
            "ocr_min_chars": ler_int_config(runtime.get("OCR_MIN_CHARS"), OCR_MIN_CHARS),
            "aws_access_key_id": runtime["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": runtime["AWS_SECRET_ACCESS_KEY"],
            "aws_region": runtime.get("AWS_REGION", "us-east-1"),
            "aws_bucket_name": runtime["AWS_BUCKET_NAME"],
        }

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
