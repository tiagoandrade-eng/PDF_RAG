from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO

import fitz
import httpx
from PIL import Image

from ..models.config import Config, TokenCounter, truncar
from ..models.constants import (
    EASYOCR_LANGS,
    FORMATOS_SUPORTADOS,
    IMAGE_AREA_MINI_THRESHOLD,
    MAX_ASPECT_RATIO,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_AREA,
    MIN_IMAGE_SIDE,
    MODEL_FULL,
    MODEL_MINI,
    OCR_MAX_SIDE,
    PAGE_RENDER_DPI,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class ImagemNormalizada:
    img_bytes: bytes
    ext: str
    width: int
    height: int
    origem: str


# ---------------------------------------------------------------------------
# Filtros de imagem
# ---------------------------------------------------------------------------
def motivo_filtro_imagem(
    width: int,
    height: int,
    min_image_area: int = MIN_IMAGE_AREA,
    min_image_side: int = MIN_IMAGE_SIDE,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> str | None:
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
        width, height,
        min_image_area=min_image_area,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
    ) is None


# ---------------------------------------------------------------------------
# Normalizacao de imagem
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Extracao de imagem
# ---------------------------------------------------------------------------
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
            width, height,
            min_image_area=min_image_area,
            min_image_side=min_image_side,
            max_aspect_ratio=max_aspect_ratio,
        )
        if motivo:
            return {
                "ok": False, "reason": f"filtered_original_{motivo}", "xref": xref,
                "width": width, "height": height, "ext": ext,
            }

    normalizada = normalizar_imagem(raw, ext) or rasterizar_xref(doc, xref)
    if not normalizada:
        logger.error("Falha total ao normalizar xref=%s.", xref)
        return {"ok": False, "reason": "normalize_error", "xref": xref, "width": width, "height": height, "ext": ext}

    motivo = motivo_filtro_imagem(
        normalizada.width, normalizada.height,
        min_image_area=min_image_area,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
    )
    if motivo:
        return {
            "ok": False, "reason": f"filtered_final_{motivo}", "xref": xref,
            "width": normalizada.width, "height": normalizada.height,
            "ext": normalizada.ext, "origem": normalizada.origem,
        }
    if normalizada.ext not in FORMATOS_SUPORTADOS:
        logger.error("Formato final invalido: ext='%s' xref=%s", normalizada.ext, xref)
        return {
            "ok": False, "reason": "invalid_format", "xref": xref,
            "width": normalizada.width, "height": normalizada.height,
            "ext": normalizada.ext, "origem": normalizada.origem,
        }

    logger.info(
        "Imagem xref=%s normalizada | origem=%s | %sx%s | ext=%s",
        xref, normalizada.origem, normalizada.width, normalizada.height, normalizada.ext,
    )
    return {
        "ok": True, "reason": "accepted", "xref": xref,
        "payload": {
            "img_bytes": normalizada.img_bytes, "ext": normalizada.ext,
            "width": normalizada.width, "height": normalizada.height,
            "renderizada": False, "origem": normalizada.origem,
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
        doc, xref,
        min_image_area=min_image_area,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
    )
    if resultado["ok"]:
        return resultado["payload"]
    return None


# ---------------------------------------------------------------------------
# EasyOCR worker (multiprocessing)
# ---------------------------------------------------------------------------
_easyocr_reader = None


def _init_easyocr_worker(langs: list[str]) -> None:
    global _easyocr_reader
    import easyocr
    _easyocr_reader = easyocr.Reader(langs, gpu=False, verbose=False)


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
            payload.get("page_number"), payload.get("img_idx"), exc,
        )
        texto = ""

    return {
        **payload,
        "texto_ocr": texto,
        "chars_uteis": len(texto.replace(" ", "").replace("\n", "")),
    }


_easyocr_worker = easyocr_worker


# ---------------------------------------------------------------------------
# OCR Service (execucao concorrente)
# ---------------------------------------------------------------------------
class OCRService:
    def __init__(self, workers: int) -> None:
        self.workers = workers

    async def executar(self, payloads: list[dict]) -> list[dict]:
        if not payloads:
            return []

        total = len(payloads)
        logger.info("EasyOCR iniciando: %s imagens | workers=%s", total, self.workers)

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        langs = payloads[0].get("langs", EASYOCR_LANGS)
        resultados: list[dict] = []
        concluidos = 0

        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_easyocr_worker,
            initargs=(langs,),
        ) as pool:
            futures = [loop.run_in_executor(pool, easyocr_worker, payload) for payload in payloads]
            for fut in asyncio.as_completed(futures):
                resultado = await fut
                resultados.append(resultado)
                concluidos += 1
                if concluidos % 10 == 0 or concluidos == total:
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "EasyOCR: %s/%s imagens (%.0f%%) | %.1fs decorridos",
                        concluidos, total, 100 * concluidos / total, elapsed,
                    )

        logger.info("EasyOCR concluido: %s imagens em %.1fs", total, time.perf_counter() - t0)
        return resultados


# ---------------------------------------------------------------------------
# Vision (OpenAI)
# ---------------------------------------------------------------------------
def escolher_modelo(width: int, height: int) -> str:
    return MODEL_FULL if (width * height) >= IMAGE_AREA_MINI_THRESHOLD else MODEL_MINI


class VisionClient:
    _MIME = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "webp": "image/webp", "gif": "image/gif",
    }
    _SYSTEM = (
        "Voce e um especialista em manuais tecnicos de manutencao e servico. "
        "Analise a imagem e forneca descricao tecnica detalhada em portugues. "
        "Identifique componentes, pecas, numeracao de referencia, setas, medidas, "
        "especificacoes e sequencia de montagem/desmontagem, se houver. "
        "Transcreva integralmente qualquer texto visivel na imagem. "
        "Seja especifico: esta descricao sera usada para busca semantica."
    )

    def __init__(self, api_key: str, counter: TokenCounter, concurrency: int) -> None:
        self._api_key = api_key
        self._counter = counter
        self._sem = asyncio.Semaphore(concurrency)

    async def descrever(
        self, session: httpx.AsyncClient, payload: dict, max_retries: int = 3
    ) -> dict:
        ext = payload.get("ext", "").lower()
        if ext not in FORMATOS_SUPORTADOS:
            return {**payload, "texto_vision": "", "modelo_usado": MODEL_FULL}

        modelo = escolher_modelo(payload.get("width", 0), payload.get("height", 0))
        mime = self._MIME[ext]
        b64 = base64.b64encode(payload["img_bytes"]).decode()
        contexto = payload.get("texto_pagina", "")[:400]
        prompt = "Descreva esta imagem tecnica detalhadamente."
        if contexto:
            prompt += f"\n\nContexto da pagina:\n{contexto}"

        body = {
            "model": modelo,
            "max_tokens": 1_000,
            "messages": [
                {"role": "system", "content": self._SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }

        async with self._sem:
            for attempt in range(max_retries):
                try:
                    resp = await session.post(
                        "https://api.openai.com/v1/chat/completions",
                        json=body,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        timeout=60.0,
                    )
                    if resp.status_code != 200:
                        logger.error(
                            "Vision HTTP %s pag %s: %s",
                            resp.status_code, payload["page_number"], resp.text[:300],
                        )
                    resp.raise_for_status()
                    resposta_json = resp.json()
                    texto = resposta_json["choices"][0]["message"]["content"].strip()
                    uso = resposta_json.get("usage", {})
                    self._counter.add(
                        modelo, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0),
                    )
                    logger.info(
                        "Vision pag %s img_%s [%s]: %s chars",
                        payload["page_number"], payload["img_idx"], modelo, len(texto),
                    )
                    return {**payload, "texto_vision": texto, "modelo_usado": modelo}
                except httpx.HTTPStatusError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        return {**payload, "texto_vision": "", "modelo_usado": modelo}
                except Exception as exc:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        logger.error(
                            "Vision falhou pag %s img_%s: %s",
                            payload["page_number"], payload["img_idx"], exc,
                        )
                        return {**payload, "texto_vision": "", "modelo_usado": modelo}

    async def processar_lote(self, payloads: list[dict]) -> list[dict]:
        async with httpx.AsyncClient() as session:
            return list(await asyncio.gather(*[self.descrever(session, payload) for payload in payloads]))


# ---------------------------------------------------------------------------
# Image pipeline (serializacao + orquestracao OCR/Vision)
# ---------------------------------------------------------------------------
def criar_metadata_imagem(payload: dict, arquivo: str, metodo: str) -> dict:
    return {
        "pagina": payload["page_number"],
        "arquivo": arquivo,
        "chunk_index": payload["chunk_index"],
        "tipo": "imagem",
        "metodo_extracao": metodo,
        "largura_px": payload["width"],
        "altura_px": payload["height"],
        "pagina_renderizada": payload["renderizada"],
        "origem": payload.get("origem", "desconhecida"),
    }


def formatar_texto_imagem(payload: dict, texto: str, metodo: str) -> str:
    label = "SCAN DE PAGINA" if payload["renderizada"] else "IMAGEM TECNICA"
    return truncar(f"[{label} - Pagina {payload['page_number']} | Metodo: {metodo}]\n{texto}")


class ImageSerializer:
    def __init__(self, cfg: Config, xrefs_vistos: set[int]) -> None:
        self.cfg = cfg
        self._xrefs_vistos = xrefs_vistos

    def serializar_paginas(
        self,
        paginas: list[tuple[int, fitz.Page, str]],
        ja_processados: set[tuple[str, str]] | None,
        stats: Counter[str] | None = None,
    ) -> list[dict]:
        payloads: list[dict] = []
        processados = ja_processados or set()

        for page_num, pag, texto_pag in paginas:
            doc = pag.parent
            imagens: list[dict] = []

            for img_info in pag.get_images(full=True):
                xref = img_info[0]
                if xref in self._xrefs_vistos:
                    if stats is not None:
                        stats["duplicate_xrefs_skipped"] += 1
                    continue
                self._xrefs_vistos.add(xref)
                if stats is not None:
                    stats["unique_xrefs_seen"] += 1
                resultado = extrair_imagem_info(
                    doc, xref,
                    min_image_area=self.cfg.min_image_area,
                    min_image_side=self.cfg.min_image_side,
                    max_aspect_ratio=self.cfg.max_aspect_ratio,
                )
                if resultado["ok"]:
                    imagens.append(resultado["payload"])
                elif stats is not None:
                    stats["images_filtered_total"] += 1
                    stats[f"filtered_reason__{resultado['reason']}"] += 1

            if not imagens and not texto_pag:
                render = renderizar_pagina(pag)
                if render:
                    if stats is not None:
                        stats["scan_pages_rendered"] += 1
                    imagens = [
                        {
                            "img_bytes": render.img_bytes, "ext": render.ext,
                            "width": render.width, "height": render.height,
                            "renderizada": True, "origem": render.origem,
                        }
                    ]
                    logger.info("Pag %s: scan detectado, renderizada em %s.", page_num, render.ext)

            for idx, img in enumerate(imagens):
                chunk_index = f"img_{idx}"
                if (str(page_num), chunk_index) in processados:
                    continue
                payloads.append(
                    {
                        "page_number": page_num, "img_idx": idx, "chunk_index": chunk_index,
                        "img_bytes": img["img_bytes"], "ext": img["ext"],
                        "width": img["width"], "height": img["height"],
                        "renderizada": img["renderizada"], "texto_pagina": texto_pag,
                        "langs": EASYOCR_LANGS, "origem": img.get("origem", "desconhecida"),
                    }
                )

        logger.info("Imagens serializadas: %s", len(payloads))
        return payloads


class ImageContentPipeline:
    def __init__(self, cfg: Config, ocr_service: OCRService, vision_client: VisionClient) -> None:
        self.cfg = cfg
        self.ocr_service = ocr_service
        self.vision_client = vision_client

    async def extrair_conteudo(self, payloads: list[dict], arquivo: str) -> tuple[list[str], list[dict]]:
        if not payloads:
            return [], []

        logger.info(
            "Pipeline imagens: %s imagem(ns) | ocr_workers=%s | vision_concurrency=%s",
            len(payloads), self.cfg.ocr_workers, self.cfg.vision_concurrency,
        )
        ocr_results = await self.ocr_service.executar(payloads)

        aceitos = [payload for payload in ocr_results if payload["chars_uteis"] >= self.cfg.ocr_min_chars]
        vision_needed = [payload for payload in ocr_results if payload["chars_uteis"] < self.cfg.ocr_min_chars]
        n_mini = sum(
            1 for payload in vision_needed if escolher_modelo(payload["width"], payload["height"]) == MODEL_MINI
        )
        logger.info(
            "EasyOCR ok: %s | gpt-4o-mini: %s | gpt-4o: %s",
            len(aceitos), n_mini, len(vision_needed) - n_mini,
        )

        vision_results: list[dict] = []
        if vision_needed:
            t1 = time.perf_counter()
            vision_results = await self.vision_client.processar_lote(vision_needed)
            logger.info("Vision: %s imagens em %.1fs", len(vision_results), time.perf_counter() - t1)

        textos: list[str] = []
        metas: list[dict] = []

        for payload in aceitos:
            textos.append(formatar_texto_imagem(payload, payload["texto_ocr"], "easyocr"))
            metas.append(criar_metadata_imagem(payload, arquivo, "easyocr"))

        for payload in vision_results:
            texto_vision = payload.get("texto_vision", "").strip()
            if not texto_vision:
                logger.warning("Vision sem resultado: pag %s img_%s. Pulando.", payload["page_number"], payload["img_idx"])
                continue
            modelo = payload.get("modelo_usado", MODEL_FULL)
            textos.append(formatar_texto_imagem(payload, texto_vision, modelo))
            metas.append(criar_metadata_imagem(payload, arquivo, modelo))

        return textos, metas
