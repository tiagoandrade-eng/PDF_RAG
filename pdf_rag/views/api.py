from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..models.config import Config, carregar_config_execucao, configure_logging
from ..controllers.processor import PDFProcessor


configure_logging()
runtime = carregar_config_execucao()

app = FastAPI(title="PDF RAG API", version="1.0.0")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ProcessarRequest(BaseModel):
    s3_pdf_key: str
    tabela: str = Field(default="minha_tabela_rag")
    db_schema: str = Field(default="public")
    processar_imagens: bool = Field(default=True)
    modo_analise: bool = Field(default=False)


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_config(req: ProcessarRequest) -> Config:
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/processar")
async def processar(req: ProcessarRequest):
    try:
        cfg = _build_config(req)
        resultado = await PDFProcessor(cfg).processar(req.s3_pdf_key, req.tabela)
        return resultado
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Configuracao ausente no .env: {exc.args[0]}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analisar")
async def analisar(req: ProcessarRequest):
    try:
        cfg = _build_config(req)
        resultado = await PDFProcessor(cfg).analisar(req.s3_pdf_key)
        return resultado
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Configuracao ausente no .env: {exc.args[0]}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
