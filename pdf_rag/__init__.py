"""
Pacote principal do projeto PDF RAG.
"""

from .cli import (
    analisar_documento_api,
    analisar_documento_async,
    main,
    processar_documento_api,
    processar_documento_async,
)
from .config import Config

__all__ = [
    "Config",
    "main",
    "analisar_documento_api",
    "analisar_documento_async",
    "processar_documento_api",
    "processar_documento_async",
]
