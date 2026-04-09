from __future__ import annotations

import logging
import os
from dataclasses import dataclass


logger = logging.getLogger(__name__)


VECTOR_DIM = 3072
BATCH_SIZES = [100, 50, 10, 1]
TAMANHO_QUEBRA_SIZES = [500, 100]
MAX_CHARS = 8_000
PAGE_RENDER_DPI = 200
MIN_IMAGE_AREA = 10_000
MIN_IMAGE_SIDE = 80
MAX_ASPECT_RATIO = 8.0
PAGE_BATCH_SIZE = 10
VISION_CONCURRENCY = 8
OCR_MIN_CHARS = 80
IMAGE_AREA_MINI_THRESHOLD = 250_000
MODEL_MINI = "gpt-4o-mini"
MODEL_FULL = "gpt-4o"
EASYOCR_LANGS = ["pt", "en"]
FORMATOS_SUPORTADOS = {"png", "jpg", "jpeg", "webp", "gif"}
MAX_IMAGE_SIDE = 2_200
DEFAULT_SCHEMA = "public"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Workers dinamicos: metade dos nucleos, minimo 1, maximo 8
OCR_WORKERS_DEFAULT = min(max(1, (os.cpu_count() or 2) // 2), 8)


@dataclass(frozen=True)
class Config:
    """
    Objeto imutavel com todas as credenciais e parametros de execucao.
    """

    openai_api_key: str
    rds_db_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    aws_bucket_name: str
    schema: str = DEFAULT_SCHEMA
    processar_imagens: bool = True
    page_batch_size: int = PAGE_BATCH_SIZE
    ocr_workers: int = OCR_WORKERS_DEFAULT
    vision_concurrency: int = VISION_CONCURRENCY
    min_image_area: int = MIN_IMAGE_AREA
    min_image_side: int = MIN_IMAGE_SIDE
    max_aspect_ratio: float = MAX_ASPECT_RATIO
    ocr_min_chars: int = OCR_MIN_CHARS
    modo_analise: bool = False


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def carregar_arquivo_env(nome_arquivo: str) -> dict[str, str]:
    caminho = os.path.join(PROJECT_ROOT, nome_arquivo)
    if not os.path.exists(caminho):
        return {}

    valores: dict[str, str] = {}
    with open(caminho, "r", encoding="utf-8-sig") as arquivo:
        for linha_num, linha in enumerate(arquivo, start=1):
            conteudo = linha.strip()
            if not conteudo or conteudo.startswith("#"):
                continue
            if "=" not in conteudo:
                logger.warning("Linha ignorada em %s:%s -> %s", nome_arquivo, linha_num, conteudo)
                continue
            chave, valor = conteudo.split("=", 1)
            chave = chave.strip()
            valor = valor.strip().strip('"').strip("'")
            if chave:
                valores[chave] = valor
    return valores


def carregar_config_execucao() -> dict[str, str]:
    config: dict[str, str] = {}
    config.update(carregar_arquivo_env(".env"))
    config.update(carregar_arquivo_env("app_settings.env"))
    config.update(os.environ)
    return config


def ler_bool_config(valor: str | None, default: bool) -> bool:
    if valor is None or not valor.strip():
        return default
    return valor.strip().lower() in {"1", "true", "t", "yes", "y", "sim", "s", "on"}


def ler_int_config(valor: str | None, default: int) -> int:
    if valor is None or not valor.strip():
        return default
    return int(valor)


def ler_float_config(valor: str | None, default: float) -> float:
    if valor is None or not valor.strip():
        return default
    return float(valor)
