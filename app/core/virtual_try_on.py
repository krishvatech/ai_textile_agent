# app/core/virtual_try_on.py
"""
Env-only Virtual Try-On (ReContext) using google-genai.

Required env:
- GOOGLE_APPLICATION_CREDENTIALS -> path to service-account JSON (e.g. D:\ai_textile_agent\credentials.json)

Optional env:
- GOOGLE_PROJECT_ID / GOOGLE_CLOUD_PROJECT  -> if absent, auto-read from the JSON's project_id
- GOOGLE_LOCATION / GOOGLE_CLOUD_LOCATION   -> default 'us-central1'
- VTO_USE_VERTEX / GOOGLE_GENAI_USE_VERTEXAI -> default True (use Vertex with ADC)
- GOOGLE_API_KEY (only if VTO_USE_VERTEX=0 / GOOGLE_GENAI_USE_VERTEXAI=false)
- VTO_MODEL            -> default 'virtual-try-on-preview-08-04'
- VTO_BASE_STEPS       -> default '60'
- VTO_SEED             -> integer; unset=random
- VTO_ADD_WATERMARK    -> '1' to enable watermark

Notes:
- With Vertex (default), auth is via ADC from GOOGLE_APPLICATION_CREDENTIALS.
- This module accepts raw bytes for both person and garment; download is handled by caller (e.g., whatsapp flow).
"""

import os
import io
import json
import tempfile
import random
import asyncio
import typing
import inspect
from uuid import uuid4
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage, ImageDraw

from google import genai
from google.genai.types import (
    HttpOptions,
    RecontextImageSource,
    RecontextImageConfig,
    ProductImage,
    Image,
)

# ---------- optional deps ----------
try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None


# ---------- utilities ----------
def _as_temp_png(data: bytes) -> str:
    p = os.path.join(tempfile.gettempdir(), f"vto_{uuid4().hex}.png")
    with open(p, "wb") as f:
        f.write(data)
    return p


def _tight_bbox_from_rgba(img_rgba: PILImage.Image):
    arr = np.array(img_rgba)
    if arr.shape[2] == 4:
        alpha = arr[..., 3]
        ys, xs = np.where(alpha > 0)
        if len(xs) and len(ys):
            return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    return None


def prep_garment_bytes(garment_bytes: bytes) -> str:
    """
    BG remove (if rembg present) + tight crop → temp PNG path.
    """
    img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")
    if rembg_remove:
        try:
            garment_bytes = rembg_remove(garment_bytes)
            img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")
        except Exception:
            # non-fatal; continue with original
            pass
    box = _tight_bbox_from_rgba(img)
    if box:
        img = img.crop(box)
    b2 = io.BytesIO()
    img.save(b2, format="PNG")
    return _as_temp_png(b2.getvalue())


def neutralize_torso_bytes(person_bytes: bytes, alpha=0.7, soften=22) -> str:
    """
    Softly mute torso so the model replaces garment cleanly.
    Falls back to no-op if mediapipe is unavailable or landmarks missing.
    """
    img = PILImage.open(io.BytesIO(person_bytes)).convert("RGBA")
    w, h = img.size
    if mp_pose is None:
        b = io.BytesIO(); img.save(b, format="PNG"); return _as_temp_png(b.getvalue())

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        rgb = np.array(img.convert("RGB"))
        res = pose.process(rgb)

    if not getattr(res, "pose_landmarks", None):
        b = io.BytesIO(); img.save(b, format="PNG"); return _as_temp_png(b.getvalue())

    lm = res.pose_landmarks.landmark
    def dn(p): return (int(p.x * w), int(p.y * h))
    # shoulders (11,12) + hips (24,23)
    pts = [dn(lm[11]), dn(lm[12]), dn(lm[24]), dn(lm[23])]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = max(0, min(xs) - 20), min(w, max(xs) + 20)
    y0, y1 = max(0, min(ys) - 30), min(h, max(ys) + 40)

    overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=soften, fill=(180, 180, 180, int(255 * alpha)))
    out = PILImage.alpha_composite(img, overlay)
    b = io.BytesIO(); out.save(b, format="PNG")
    return _as_temp_png(b.getvalue())


# ---------- env helpers ----------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1", "true", "t", "yes", "y")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _project_from_adc() -> typing.Optional[str]:
    adc = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not adc or not os.path.isfile(adc):
        return None
    try:
        with open(adc, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("project_id")
    except Exception:
        return None


# ---------- config ----------
@dataclass
class VTOConfig:
    base_steps: int = _env_int("VTO_BASE_STEPS", 60)
    seed: typing.Optional[int] = (int(os.getenv("VTO_SEED")) if os.getenv("VTO_SEED") else None)
    add_watermark: bool = _env_bool("VTO_ADD_WATERMARK", False)
    model: str = os.getenv("VTO_MODEL", "virtual-try-on-preview-08-04")

    # Vertex by default; allow Google-provided env names too
    use_vertex: bool = _env_bool("VTO_USE_VERTEX", _env_bool("GOOGLE_GENAI_USE_VERTEXAI", True))
    project: typing.Optional[str] = (
        os.getenv("GOOGLE_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or _project_from_adc()
    )
    location: str = (
        os.getenv("GOOGLE_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or "us-central1"
    )


def _validate_adc_or_die():
    adc = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not adc:
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
    if not os.path.isfile(adc):
        raise EnvironmentError(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {adc}")


def _make_client(cfg: VTOConfig) -> genai.Client:
    if cfg.use_vertex:
        _validate_adc_or_die()
        if not cfg.project:
            raise EnvironmentError(
                "GOOGLE_PROJECT_ID/GOOGLE_CLOUD_PROJECT not set and project_id not found in credentials JSON."
            )
        return genai.Client(
            vertexai=True,
            project=cfg.project,
            location=cfg.location,
            http_options=HttpOptions(api_version="v1"),
        )
    # Non-Vertex path
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("VTO_USE_VERTEX=0 requires GOOGLE_API_KEY in the environment.")
    return genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))


# ---------- main API ----------
async def generate_vto_image(
    person_bytes: bytes,
    garment_bytes: bytes,
    cfg: typing.Optional[VTOConfig] = None,
) -> bytes:
    """
    Returns PNG bytes of the try-on result. Fully env-driven; no hardcoded paths/keys.
    """
    cfg = cfg or VTOConfig()
    client = _make_client(cfg)

    # Prep inputs
    person_tmp = neutralize_torso_bytes(person_bytes)
    garment_tmp = prep_garment_bytes(garment_bytes)

    person_img = Image.from_file(location=person_tmp)
    garment_img = Image.from_file(location=garment_tmp)

    # Recontext config
    try:
        re_cfg = RecontextImageConfig(
            number_of_images=1,
            base_steps=cfg.base_steps,
            seed=(cfg.seed or random.randint(1, 10_000_000)),
            add_watermark=cfg.add_watermark,
        )
    except Exception:
        # In case of older SDK lacking some fields
        re_cfg = RecontextImageConfig(number_of_images=1)

    # Build source once
    src = RecontextImageSource(
        person_image=person_img,
        product_images=[ProductImage(product_image=garment_img)],
    )

    # ---- Robust call across SDK versions ----
    def _call_recontext():
        fn = client.models.recontext_image
        params = tuple(inspect.signature(fn).parameters.keys())

        # Preferred (newer SDKs)
        if "model" in params and "source" in params:
            return fn(model=cfg.model, source=src, config=re_cfg)

        # Some early previews used 'image' instead of 'source'
        if "model" in params and "image" in params:
            return fn(model=cfg.model, image=src, config=re_cfg)

        # Fallback to positional (older, undocumented shapes)
        try:
            return fn(cfg.model, src, re_cfg)
        except TypeError:
            # last resort: try without config
            return fn(cfg.model, src)

    resp = await asyncio.to_thread(_call_recontext)

    # Extract first image
    images = getattr(resp, "generated_images", None) or getattr(resp, "images", None) or []
    if not images:
        raise RuntimeError("VTO returned no image")

    out = images[0].image if hasattr(images[0], "image") else images[0]

    # Normalize to PNG bytes
    if isinstance(out, PILImage.Image):
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(out, bytes):
        return out
    if hasattr(out, "save"):
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    raise RuntimeError("Unsupported VTO output type")