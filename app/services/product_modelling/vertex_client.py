from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional

from google import genai
from google.genai.types import (
    HttpOptions,
    Image,
    ProductImage,
    RecontextImageConfig,
    RecontextImageSource,
)

try:
    from google.genai.types import Part
except Exception:
    Part = None  # type: ignore

log = logging.getLogger(__name__)


def _require_adc():
    adc = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not adc:
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
    if not os.path.isfile(adc):
        raise EnvironmentError(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {adc}")


def _project_and_location() -> tuple[str, str]:
    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GOOGLE_PROJECT_ID")
        or os.getenv("GOOGLE_PROJECT")
    )
    location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_LOCATION") or "us-central1"
    if not project:
        raise EnvironmentError("GOOGLE_CLOUD_PROJECT/GOOGLE_PROJECT_ID is not set.")
    return project, location


def _make_client() -> genai.Client:
    _require_adc()
    project, location = _project_and_location()
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )


def _as_temp_png(data: bytes) -> str:
    path = os.path.join(tempfile.gettempdir(), f"pm_{os.getpid()}_{os.urandom(6).hex()}.png")
    with open(path, "wb") as f:
        f.write(data)
    return path


def _image_from_bytes(data: bytes) -> Image:
    return Image.from_file(location=_as_temp_png(data))


def _extract_text(resp: Any) -> str:
    txt = getattr(resp, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t.strip():
                return t
    if isinstance(resp, dict):
        for k in ("text", "output", "content"):
            if isinstance(resp.get(k), str) and resp[k].strip():
                return resp[k]
    return ""


def _safe_json(text: str) -> Dict[str, str]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items()}
    except Exception:
        pass
    return {
        "lighting": "",
        "background": "",
        "mood": "",
        "camera": "",
        "composition": "",
        "colors": "",
        "notes": text.strip()[:800] if text else "",
    }


def _caption_prompt() -> str:
    return (
        "Analyze the reference image style and return JSON with keys: "
        "lighting, background, mood, camera, composition, colors, notes. "
        "Keep values short phrases."
    )


def _build_parts(prompt: str, image_bytes: bytes) -> List[Any]:
    if Part is not None:
        return [Part.from_text(prompt), Part.from_bytes(image_bytes, mime_type="image/png")]
    # Fallback to dict-style parts if Part is unavailable
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": b64}},
    ]


async def _call_generate_content(model: str, contents: Iterable[Any], timeout_s: float) -> Any:
    client = _make_client()
    fn = client.models.generate_content
    params = inspect.signature(fn).parameters
    kwargs: Dict[str, Any] = {"model": model, "contents": list(contents)}
    if "config" in params:
        kwargs["config"] = {
            "temperature": 0,
            "response_mime_type": "application/json",
        }
    return await asyncio.wait_for(asyncio.to_thread(fn, **kwargs), timeout=timeout_s)


async def caption_reference_style(reference_image_bytes: bytes, timeout_s: float = 25.0) -> Dict[str, str]:
    prompt = _caption_prompt()
    contents = _build_parts(prompt, reference_image_bytes)
    caption_model = os.getenv("IMAGEN_CAPTION_MODEL", "imagen-3.0-caption-001")
    try:
        resp = await _call_generate_content(caption_model, contents, timeout_s)
        return _safe_json(_extract_text(resp))
    except Exception as e:
        log.debug("Imagen caption failed, falling back to Gemini: %s", e)
    gemini_model = os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash")
    resp = await _call_generate_content(gemini_model, contents, timeout_s)
    return _safe_json(_extract_text(resp))


def _decode_base64_field(val: Any) -> Optional[bytes]:
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, str):
        try:
            return base64.b64decode(val)
        except Exception:
            return None
    return None


def _to_png_bytes_from_unknown(obj: Any) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, Image):
        for attr in ("bytes", "image_bytes", "data", "content"):
            val = getattr(obj, attr, None)
            if isinstance(val, (bytes, bytearray)):
                return bytes(val)
            if isinstance(val, str):
                decoded = _decode_base64_field(val)
                if decoded:
                    return decoded
    if isinstance(obj, dict):
        for k in ("bytesBase64Encoded", "imageBytes", "image", "content"):
            if k in obj:
                b = _decode_base64_field(obj[k])
                if b:
                    return b
    if isinstance(obj, (list, tuple)):
        for item in obj:
            try:
                return _to_png_bytes_from_unknown(item)
            except Exception:
                continue
    raise RuntimeError("Unsupported image output type")


async def product_recontext(
    product_image_bytes: bytes,
    prompt: str,
    num_images: int,
    seeds: Optional[List[int]] = None,
    timeout_s: float = 90.0,
) -> List[bytes]:
    client = _make_client()
    model = os.getenv("PRODUCT_RECONTEXT_MODEL", "product-recontext-preview-09-30")
    seed = (seeds or [None])[0]
    try:
        cfg = RecontextImageConfig(
            number_of_images=num_images,
            seed=seed,
            add_watermark=False,
            prompt=prompt,
        )
    except Exception:
        cfg = RecontextImageConfig(number_of_images=num_images)

    src = RecontextImageSource(product_images=[ProductImage(product_image=_image_from_bytes(product_image_bytes))])

    fn = client.models.recontext_image
    params = inspect.signature(fn).parameters
    kwargs: Dict[str, Any] = {"model": model}
    if "source" in params:
        kwargs["source"] = src
    elif "image" in params:
        kwargs["image"] = src
    if "config" in params:
        kwargs["config"] = cfg
    if "prompt" in params:
        kwargs["prompt"] = prompt

    resp = await asyncio.wait_for(asyncio.to_thread(fn, **kwargs), timeout=timeout_s)
    images = (
        getattr(resp, "generated_images", None)
        or getattr(resp, "images", None)
        or getattr(resp, "predictions", None)
        or (resp if isinstance(resp, (list, tuple)) else None)
        or []
    )
    if not images and isinstance(resp, dict):
        images = resp.get("predictions", [])
    out: List[bytes] = []
    for cand in images:
        obj = cand.image if hasattr(cand, "image") else cand
        out.append(_to_png_bytes_from_unknown(obj))
    return out


async def edit_place_product_on_person(
    person_image_bytes: bytes,
    product_image_bytes: bytes,
    prompt: str,
    num_images: int,
    timeout_s: float = 90.0,
) -> List[bytes]:
    client = _make_client()
    model = os.getenv("IMAGEN_EDIT_MODEL", "imagen-3.0-edit-002")
    fn = getattr(client.models, "edit_image", None)
    if fn is None:
        return await _recontext_on_person(
            person_image_bytes=person_image_bytes,
            product_image_bytes=product_image_bytes,
            prompt=prompt,
            num_images=num_images,
            timeout_s=timeout_s,
        )
    params = inspect.signature(fn).parameters
    kwargs: Dict[str, Any] = {"model": model}
    person_img = _image_from_bytes(person_image_bytes)
    product_img = _image_from_bytes(product_image_bytes)

    if "image" in params:
        kwargs["image"] = person_img
    if "prompt" in params:
        kwargs["prompt"] = prompt
    if "reference_images" in params:
        kwargs["reference_images"] = [product_img]
    elif "reference_image" in params:
        kwargs["reference_image"] = product_img
    elif "product_image" in params:
        kwargs["product_image"] = product_img

    if "config" in params:
        kwargs["config"] = {"number_of_images": num_images}
    elif "number_of_images" in params:
        kwargs["number_of_images"] = num_images

    resp = await asyncio.wait_for(asyncio.to_thread(fn, **kwargs), timeout=timeout_s)
    images = (
        getattr(resp, "generated_images", None)
        or getattr(resp, "images", None)
        or getattr(resp, "predictions", None)
        or (resp if isinstance(resp, (list, tuple)) else None)
        or []
    )
    if not images and isinstance(resp, dict):
        images = resp.get("predictions", [])
    out: List[bytes] = []
    for cand in images:
        obj = cand.image if hasattr(cand, "image") else cand
        out.append(_to_png_bytes_from_unknown(obj))
    return out


async def _recontext_on_person(
    person_image_bytes: bytes,
    product_image_bytes: bytes,
    prompt: str,
    num_images: int,
    timeout_s: float,
) -> List[bytes]:
    client = _make_client()
    model = os.getenv("PRODUCT_RECONTEXT_MODEL", "product-recontext-preview-09-30")
    seed = None
    try:
        cfg = RecontextImageConfig(
            number_of_images=num_images,
            seed=seed,
            add_watermark=False,
            prompt=prompt,
        )
    except Exception:
        cfg = RecontextImageConfig(number_of_images=num_images)

    src = RecontextImageSource(
        person_image=_image_from_bytes(person_image_bytes),
        product_images=[ProductImage(product_image=_image_from_bytes(product_image_bytes))],
    )

    fn = client.models.recontext_image
    params = inspect.signature(fn).parameters
    kwargs: Dict[str, Any] = {"model": model}
    if "source" in params:
        kwargs["source"] = src
    elif "image" in params:
        kwargs["image"] = src
    if "config" in params:
        kwargs["config"] = cfg
    if "prompt" in params:
        kwargs["prompt"] = prompt

    resp = await asyncio.wait_for(asyncio.to_thread(fn, **kwargs), timeout=timeout_s)
    images = (
        getattr(resp, "generated_images", None)
        or getattr(resp, "images", None)
        or getattr(resp, "predictions", None)
        or (resp if isinstance(resp, (list, tuple)) else None)
        or []
    )
    if not images and isinstance(resp, dict):
        images = resp.get("predictions", [])
    out: List[bytes] = []
    for cand in images:
        obj = cand.image if hasattr(cand, "image") else cand
        out.append(_to_png_bytes_from_unknown(obj))
    return out
