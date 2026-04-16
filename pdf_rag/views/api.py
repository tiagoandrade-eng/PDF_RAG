from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from ..models.config import Config, carregar_config_execucao, configure_logging
from ..controllers.processor import ConnectivityError, PDFProcessor
from ..models.database import DatabaseReadError


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF RAG API", version="1.0.0")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class ProcessarRequest(BaseModel):
    s3_pdf_key: str
    tabela: str = Field(default="minha_tabela_rag")
    db_schema: str = Field(default="public")
    processar_imagens: bool = Field(default=True)
    modo_analise: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"


class PersistenciaResponse(BaseModel):
    attempted: int
    inserted: int
    failed: int


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ProcessarResponse(BaseModel):
    status: str
    schema_: str = Field(alias="schema")
    tabela: str
    chunks_texto: int
    chunks_imagem: int
    chunks_pulados: int
    persistencia: PersistenciaResponse
    openai_tokens: dict[str, TokenUsage]
    openai_custo_usd: float


class VisionModelSplit(BaseModel):
    gpt_4o_mini: int = Field(alias="gpt-4o-mini")
    gpt_4o: int = Field(alias="gpt-4o")


class FiltrosResponse(BaseModel):
    min_image_area: int
    min_image_side: int
    max_aspect_ratio: float


class AnalisarResponse(BaseModel):
    status: str
    arquivo: str
    pages_total: int
    pages_with_text: int
    pages_without_text: int
    avg_text_chars_per_page: float
    image_occurrences_total: int
    unique_xrefs_seen: int
    duplicate_xrefs_skipped: int
    images_filtered_total: int
    images_filtered_by_reason: dict[str, int]
    scan_pages_rendered: int
    images_to_ocr: int
    ocr_accepted: int
    vision_needed: int
    vision_model_split: VisionModelSplit
    ocr_min_chars: int
    filters: FiltrosResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_config(req: ProcessarRequest) -> Config:
    runtime = carregar_config_execucao()
    return Config(
        openai_api_key=runtime.get("OPENAI_API_KEY", ""),
        rds_db_url=runtime.get("RDS_DB_URL", ""),
        aws_access_key_id=runtime["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=runtime["AWS_SECRET_ACCESS_KEY"],
        aws_region=runtime.get("AWS_REGION", "us-east-1"),
        aws_bucket_name=runtime["AWS_BUCKET_NAME"],
        db_schema=req.db_schema,
        processar_imagens=req.processar_imagens,
        modo_analise=req.modo_analise,
    )


def _formatar_erros_validacao(exc: ValidationError) -> str:
    erros = []
    for err in exc.errors():
        campo = ".".join(str(loc) for loc in err.get("loc", []))
        if campo:
            erros.append(f"{campo}: {err['msg']}")
        else:
            erros.append(err["msg"])
    return "Configuracao invalida no .env: " + "; ".join(erros)


def _formatar_erros_conectividade(exc: ConnectivityError) -> dict:
    return {
        "mensagem": "Falha no preflight de conexoes. Verifique credenciais e disponibilidade dos servicos.",
        "erros": exc.errors,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/processar", response_model=ProcessarResponse)
async def processar(req: ProcessarRequest):
    try:
        cfg = _build_config(req)
        processor = PDFProcessor(cfg)
        await processor.validar_conexoes(checar_db=True, checar_openai=True)
        resultado = await processor.processar(req.s3_pdf_key, req.tabela)
        return resultado
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Configuracao ausente no .env: {exc.args[0]}")
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=_formatar_erros_validacao(exc))
    except ConnectivityError as exc:
        raise HTTPException(status_code=503, detail=_formatar_erros_conectividade(exc))
    except DatabaseReadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Erro inesperado em /processar: %s", exc)
        raise HTTPException(status_code=500, detail="Erro interno ao processar requisicao.")


@app.post("/analisar", response_model=AnalisarResponse)
async def analisar(req: ProcessarRequest):
    try:
        cfg = _build_config(req)
        processor = PDFProcessor(cfg)
        await processor.validar_conexoes(checar_db=False, checar_openai=False)
        resultado = await processor.analisar(req.s3_pdf_key)
        return resultado
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Configuracao ausente no .env: {exc.args[0]}")
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=_formatar_erros_validacao(exc))
    except ConnectivityError as exc:
        raise HTTPException(status_code=503, detail=_formatar_erros_conectividade(exc))
    except DatabaseReadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Erro inesperado em /analisar: %s", exc)
        raise HTTPException(status_code=500, detail="Erro interno ao processar requisicao.")
