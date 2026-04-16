from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Dimensoes e limites
# ---------------------------------------------------------------------------
VECTOR_DIM = 3072
MAX_CHARS = 8_000
MAX_IMAGE_SIDE = 2_200
OCR_MAX_SIDE = 1200
PAGE_RENDER_DPI = 200

# ---------------------------------------------------------------------------
# Filtros de imagem
# ---------------------------------------------------------------------------
MIN_IMAGE_AREA = 10_000
MIN_IMAGE_SIDE = 80
MAX_ASPECT_RATIO = 8.0
IMAGE_AREA_MINI_THRESHOLD = 250_000

# ---------------------------------------------------------------------------
# Batch e concorrencia
# ---------------------------------------------------------------------------
BATCH_SIZES = [100, 50, 10, 1]
TAMANHO_QUEBRA_SIZES = [500, 100]
PAGE_BATCH_SIZE = 100
VISION_CONCURRENCY = 6
OCR_MIN_CHARS = 60
OCR_WORKERS_DEFAULT = min(max(1, (os.cpu_count() or 2) // 2), 8)

# ---------------------------------------------------------------------------
# Modelos OpenAI
# ---------------------------------------------------------------------------
MODEL_MINI = "gpt-4o-mini"
MODEL_FULL = "gpt-4o"

# ---------------------------------------------------------------------------
# Formatos e idiomas
# ---------------------------------------------------------------------------
EASYOCR_LANGS = ["pt", "en"]
FORMATOS_SUPORTADOS = {"png", "jpg", "jpeg", "webp", "gif"}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SCHEMA = "public"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
