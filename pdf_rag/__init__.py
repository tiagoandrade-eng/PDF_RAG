"""
Pacote principal do projeto PDF RAG.
"""

from .models.config import Config
from .views.api import app

__all__ = ["Config", "app"]
