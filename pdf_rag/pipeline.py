from __future__ import annotations

"""
Modulo de compatibilidade.

Mantem imports antigos funcionando enquanto a implementacao principal
fica distribuida em modulos menores dentro do pacote.
"""

from .analysis_service import AnalysisService
from .cli import (
    analisar_documento_api,
    analisar_documento_async,
    main,
    processar_documento_api,
    processar_documento_async,
)
from .config import Config
from .database import DatabaseManager, normalizar_nome_tabela
from .embeddings import EmbeddingClient, truncar
from .image_pipeline import ImageContentPipeline, ImageSerializer
from .images import ImagemNormalizada, _easyocr_worker, easyocr_worker, extrair_imagem_safe, normalizar_imagem
from .images import rasterizar_xref, renderizar_pagina
from .ocr_service import OCRService
from .processor import PDFProcessor
from .text_pipeline import limpar_texto_pagina, serializar_texto_pagina
from .tokens import TokenCounter
from .vision import VisionClient, escolher_modelo

__all__ = [
    "Config",
    "TokenCounter",
    "DatabaseManager",
    "EmbeddingClient",
    "ImagemNormalizada",
    "VisionClient",
    "PDFProcessor",
    "OCRService",
    "ImageSerializer",
    "ImageContentPipeline",
    "AnalysisService",
    "normalizar_nome_tabela",
    "truncar",
    "escolher_modelo",
    "limpar_texto_pagina",
    "serializar_texto_pagina",
    "normalizar_imagem",
    "rasterizar_xref",
    "renderizar_pagina",
    "extrair_imagem_safe",
    "easyocr_worker",
    "_easyocr_worker",
    "analisar_documento_async",
    "analisar_documento_api",
    "processar_documento_async",
    "processar_documento_api",
    "main",
]
