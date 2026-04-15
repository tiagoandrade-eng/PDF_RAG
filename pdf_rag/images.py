from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO

import fitz
from PIL import Image

from .config import FORMATOS_SUPORTADOS, MAX_ASPECT_RATIO, MAX_IMAGE_SIDE, MIN_IMAGE_AREA, MIN_IMAGE_SIDE, PAGE_RENDER_DPI


logger = logging.getLogger(__name__)


@dataclass
class ImagemNormalizada:
    img_bytes: bytes
    ext: str
    width: int
    height: int
    origem: str


def motivo_filtro_imagem(
    width: int,
    height: int,
    min_image_area: int = MIN_IMAGE_AREA,
    min_image_side: int = MIN_IMAGE_SIDE,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> str | None:
    """Retorna o motivo do descarte quando a imagem parece decorativa."""
    if width * height < min_image_area:
        return "area"
    if width < min_image_side or height < min_image_side:
        return "side"
    ratio = max(width, height) / max(min(width, height), 1)
    if ratio > max_aspect_ratio:
        return "aspect_ratio"
    return None


def _imagem_util(
    width: int,
    height: int,
    min_image_area: int = MIN_IMAGE_AREA,
    min_image_side: int = MIN_IMAGE_SIDE,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> bool:
    return motivo_filtro_imagem(
        width,
        height,
        min_image_area=min_image_area,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
    ) is None


def _redimensionar(img: Image.Image) -> Image.Image:
    width, height = img.size
    maior_lado = max(width, height)
    if maior_lado <= MAX_IMAGE_SIDE:
        return img

    scale = MAX_IMAGE_SIDE / maior_lado
    return img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)


def _pil_para_bytes(img: Image.Image, prefer_png: bool = True) -> tuple[bytes, str, int, int]:
    img = _redimensionar(img)
    buffer = BytesIO()

    if prefer_png:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        img.save(buffer, format="PNG", optimize=True)
        ext = "png"
    else:
        img.convert("RGB").save(buffer, format="JPEG", quality=90)
        ext = "jpeg"

    return buffer.getvalue(), ext, img.width, img.height


def normalizar_imagem(raw: bytes, ext: str) -> ImagemNormalizada | None:
    ext = (ext or "").lower()

    if ext in FORMATOS_SUPORTADOS:
        try:
            with Image.open(BytesIO(raw)) as img:
                img.load()
                data, final_ext, width, height = _pil_para_bytes(img)
                return ImagemNormalizada(data, final_ext, width, height, "original")
        except Exception:
            logger.warning("Formato '%s' falhou na validacao. Tentando via PIL.", ext)

    try:
        with Image.open(BytesIO(raw)) as img:
            img.load()
            data, final_ext, width, height = _pil_para_bytes(img)
            return ImagemNormalizada(data, final_ext, width, height, "pil")
    except Exception as exc:
        logger.warning("Falha PIL (ext='%s'): %s", ext, exc)
        return None


def rasterizar_xref(doc: fitz.Document, xref: int) -> ImagemNormalizada | None:
    try:
        pix = fitz.Pixmap(doc, xref)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)

        with Image.open(BytesIO(pix.tobytes("png"))) as img:
            data, ext, width, height = _pil_para_bytes(img)

        return ImagemNormalizada(data, ext, width, height, "pymupdf")
    except Exception as exc:
        logger.warning("Falha rasterizacao xref=%s: %s", xref, exc)
        return None


def renderizar_pagina(pagina: fitz.Page) -> ImagemNormalizada | None:
    try:
        matrix = fitz.Matrix(PAGE_RENDER_DPI / 72, PAGE_RENDER_DPI / 72)
        pix = pagina.get_pixmap(matrix=matrix, alpha=False)

        with Image.open(BytesIO(pix.tobytes("png"))) as img:
            data, ext, width, height = _pil_para_bytes(img, prefer_png=False)

        return ImagemNormalizada(data, ext, width, height, "render")
    except Exception as exc:
        logger.warning("Falha ao renderizar pag %s: %s", pagina.number + 1, exc)
        return None


def extrair_imagem_info(
    doc: fitz.Document,
    xref: int,
    min_image_area: int = MIN_IMAGE_AREA,
    min_image_side: int = MIN_IMAGE_SIDE,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> dict:
    try:
        base_img = doc.extract_image(xref)
    except Exception as exc:
        logger.warning("Falha extract_image xref=%s: %s", xref, exc)
        return {"ok": False, "reason": "extract_error", "xref": xref}

    raw = base_img.get("image")
    ext = (base_img.get("ext") or "").lower()
    width = int(base_img.get("width") or 0)
    height = int(base_img.get("height") or 0)

    if not raw:
        return {"ok": False, "reason": "empty_image", "xref": xref}
    if width > 0 and height > 0:
        motivo = motivo_filtro_imagem(
            width,
            height,
            min_image_area=min_image_area,
            min_image_side=min_image_side,
            max_aspect_ratio=max_aspect_ratio,
        )
        if motivo:
            return {
                "ok": False,
                "reason": f"filtered_original_{motivo}",
                "xref": xref,
                "width": width,
                "height": height,
                "ext": ext,
            }

    normalizada = normalizar_imagem(raw, ext) or rasterizar_xref(doc, xref)
    if not normalizada:
        logger.error("Falha total ao normalizar xref=%s.", xref)
        return {"ok": False, "reason": "normalize_error", "xref": xref, "width": width, "height": height, "ext": ext}

    motivo = motivo_filtro_imagem(
        normalizada.width,
        normalizada.height,
        min_image_area=min_image_area,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
    )
    if motivo:
        return {
            "ok": False,
            "reason": f"filtered_final_{motivo}",
            "xref": xref,
            "width": normalizada.width,
            "height": normalizada.height,
            "ext": normalizada.ext,
            "origem": normalizada.origem,
        }
    if normalizada.ext not in FORMATOS_SUPORTADOS:
        logger.error("Formato final invalido: ext='%s' xref=%s", normalizada.ext, xref)
        return {
            "ok": False,
            "reason": "invalid_format",
            "xref": xref,
            "width": normalizada.width,
            "height": normalizada.height,
            "ext": normalizada.ext,
            "origem": normalizada.origem,
        }

    logger.info(
        "Imagem xref=%s normalizada | origem=%s | %sx%s | ext=%s",
        xref,
        normalizada.origem,
        normalizada.width,
        normalizada.height,
        normalizada.ext,
    )
    return {
        "ok": True,
        "reason": "accepted",
        "xref": xref,
        "payload": {
            "img_bytes": normalizada.img_bytes,
            "ext": normalizada.ext,
            "width": normalizada.width,
            "height": normalizada.height,
            "renderizada": False,
            "origem": normalizada.origem,
        },
    }


def extrair_imagem_safe(
    doc: fitz.Document,
    xref: int,
    min_image_area: int = MIN_IMAGE_AREA,
    min_image_side: int = MIN_IMAGE_SIDE,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> dict | None:
    resultado = extrair_imagem_info(
        doc,
        xref,
        min_image_area=min_image_area,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
    )
    if resultado["ok"]:
        return resultado["payload"]
    return None


_easyocr_reader = None


def _init_easyocr_worker(langs: list[str]) -> None:
    global _easyocr_reader
    import easyocr

    _easyocr_reader = easyocr.Reader(langs, gpu=False, verbose=False)


OCR_MAX_SIDE = 1200


def easyocr_worker(payload: dict) -> dict:
    import numpy as np

    try:
        reader = _easyocr_reader
        if reader is None:
            import easyocr

            reader = easyocr.Reader(payload.get("langs", ["pt", "en"]), gpu=False, verbose=False)
        img = Image.open(BytesIO(payload["img_bytes"])).convert("RGB")
        maior_lado = max(img.width, img.height)
        if maior_lado > OCR_MAX_SIDE:
            scale = OCR_MAX_SIDE / maior_lado
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
        img_arr = np.array(img)
        results = reader.readtext(img_arr, detail=1, paragraph=False)
        texto = " ".join(r[1] for r in results if r[2] >= 0.3).strip()
    except Exception as exc:
        logger.warning(
            "EasyOCR falhou na pagina %s img_%s: %s",
            payload.get("page_number"),
            payload.get("img_idx"),
            exc,
        )
        texto = ""

    return {
        **payload,
        "texto_ocr": texto,
        "chars_uteis": len(texto.replace(" ", "").replace("\n", "")),
    }


_easyocr_worker = easyocr_worker
