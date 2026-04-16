from __future__ import annotations

import logging
import os
import threading

from pydantic import BaseModel, Field

from .constants import (
    DEFAULT_SCHEMA,
    MAX_ASPECT_RATIO,
    MAX_CHARS,
    MIN_IMAGE_AREA,
    MIN_IMAGE_SIDE,
    OCR_MIN_CHARS,
    OCR_WORKERS_DEFAULT,
    PAGE_BATCH_SIZE,
    PROJECT_ROOT,
    VISION_CONCURRENCY,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class Config(BaseModel):
    model_config = {"frozen": True}

    openai_api_key: str
    rds_db_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    aws_bucket_name: str
    db_schema: str = Field(default=DEFAULT_SCHEMA)
    processar_imagens: bool = Field(default=True)
    page_batch_size: int = Field(default=PAGE_BATCH_SIZE, gt=0)
    ocr_workers: int = Field(default=OCR_WORKERS_DEFAULT, gt=0)
    vision_concurrency: int = Field(default=VISION_CONCURRENCY, gt=0)
    min_image_area: int = Field(default=MIN_IMAGE_AREA, ge=0)
    min_image_side: int = Field(default=MIN_IMAGE_SIDE, ge=0)
    max_aspect_ratio: float = Field(default=MAX_ASPECT_RATIO, gt=0)
    ocr_min_chars: int = Field(default=OCR_MIN_CHARS, ge=0)
    modo_analise: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Leitura de .env
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Truncar texto (usado por images e processor)
# ---------------------------------------------------------------------------
def truncar(texto: str) -> str:
    if len(texto) > MAX_CHARS:
        logger.warning("Texto truncado: %s -> %s chars.", len(texto), MAX_CHARS)
        return texto[:MAX_CHARS]
    return texto


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------
class TokenCounter:
    PRECOS = {
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
        "gpt-4o": {"input": 0.0025, "output": 0.010},
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tokens: dict[str, dict[str, int]] = {}

    def add(self, modelo: str, input_tokens: int, output_tokens: int = 0) -> None:
        with self._lock:
            entry = self.tokens.setdefault(modelo, {"input": 0, "output": 0})
            entry["input"] += input_tokens
            entry["output"] += output_tokens

    def custo_usd(self) -> float:
        total = 0.0
        for modelo, contagem in self.tokens.items():
            preco = self.PRECOS.get(modelo, {"input": 0.0, "output": 0.0})
            total += (contagem["input"] / 1000) * preco["input"]
            total += (contagem["output"] / 1000) * preco.get("output", 0.0)
        return total

    def resumo(self) -> str:
        if not self.tokens:
            return "   Tokens: nenhuma chamada registrada"

        linhas = ["   -- Tokens OpenAI ---------------------"]
        total_input = 0
        total_output = 0

        for modelo, contagem in sorted(self.tokens.items()):
            input_tokens = contagem["input"]
            output_tokens = contagem["output"]
            total_input += input_tokens
            total_output += output_tokens
            preco = self.PRECOS.get(modelo, {"input": 0.0, "output": 0.0})
            custo = (input_tokens / 1000) * preco["input"]
            custo += (output_tokens / 1000) * preco.get("output", 0.0)
            linhas.append(
                f"   {modelo:<28} in={input_tokens:>8,}  out={output_tokens:>7,}  ~${custo:.4f}"
            )

        linhas.append(
            f"   {'TOTAL':<28} in={total_input:>8,}  out={total_output:>7,}  ~${self.custo_usd():.4f}"
        )
        linhas.append("   ----------------------------------------")
        return "\n".join(linhas)
