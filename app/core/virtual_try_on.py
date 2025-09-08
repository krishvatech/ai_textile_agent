# app/core/virtual_try_on.py
"""
Env-only Virtual Try-On (ReContext) using google-genai.

Required env:
- GOOGLE_APPLICATION_CREDENTIALS -> path to service-account JSON (e.g. D:\ai_textile_agent\credentials.json)

Optional env:
- GOOGLE_PROJECT_ID / GOOGLE_CLOUD_PROJECT  -> if absent, auto-read from the JSON's project_id
- GOOGLE_LOCATION / GOOGLE_CLOUD_LOCATION   -> set to your Vertex location (e.g. 'us-central1')
- VTO_USE_VERTEX / GOOGLE_GENAI_USE_VERTEXAI -> default True (use Vertex with ADC)
- GOOGLE_API_KEY (only if VTO_USE_VERTEX=0 / GOOGLE_GENAI_USE_VERTEXAI=false)
- VTO_MODEL            -> default 'virtual-try-on-preview-08-04'
- VTO_BASE_STEPS       -> default '70'
- VTO_SEED             -> integer; unset=random
- VTO_ADD_WATERMARK    -> '1' to enable watermark
- VTO_NEUTRALIZE_TORSO -> default '1' (enable soft neutralization)
- VTO_TORSO_ALPHA      -> default 0.70 (how strong to dim inside mask)
- VTO_TORSO_SOFTEN     -> default 22  (kept for compatibility; not box radius anymore)
- VTO_GARMENT_MAX_SIDE -> default 1600
- VTO_NUM_IMAGES       -> default 4
- VTO_IS_FLARE         -> '1' to keep full silhouette + padding for lehenga/gown/anarkali
"""

import os
import io
import json
import base64
import tempfile
import random
import asyncio
import typing
import inspect
import logging
from uuid import uuid4
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFilter, ImageEnhance

from google import genai
from google.genai.types import (
    HttpOptions,
    RecontextImageSource,
    RecontextImageConfig,
    ProductImage,
    Image,  # google.genai.types.Image (not PIL)
)

log = logging.getLogger(__name__)

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


def _pad_canvas(img: PILImage.Image, pad_w=180, pad_h=200) -> PILImage.Image:
    """Pad around the garment so flared hems aren’t clipped after bg-removal."""
    W = img.width + pad_w * 2
    H = img.height + pad_h
    canvas = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(img, (pad_w, 0))
    return canvas


def prep_garment_bytes(garment_bytes: bytes) -> str:
    """
    BG remove (if rembg present) + (tight crop OR keep full silhouette for flare)
    + safe max-side → temp PNG path.
    """
    img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")

    # optional background remove
    if rembg_remove:
        try:
            garment_bytes = rembg_remove(garment_bytes)
            img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")
        except Exception as e:
            log.debug("rembg failed (non-fatal): %s", e)

    is_flare = _env_bool("VTO_IS_FLARE", False)

    if is_flare:
        # keep full silhouette; just pad so wide hems are preserved
        img = _pad_canvas(img, pad_w=200, pad_h=260)
    else:
        # tight bbox from alpha (good for straight-fall items like saree)
        box = _tight_bbox_from_rgba(img)
        if box:
            img = img.crop(box)

    # safe max-side (prevents extreme scales that hurt warping)
    max_side = _env_int("VTO_GARMENT_MAX_SIDE", 1600)
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

    b2 = io.BytesIO()
    img.save(b2, format="PNG")
    return _as_temp_png(b2.getvalue())


def neutralize_torso_bytes(
    person_bytes: bytes,
    alpha=0.7,
    soften=22,  # kept for env compatibility; not used as a hard rectangle radius
) -> typing.Union[str, typing.Tuple[str, typing.Tuple[int, int, int, int]]]:
    """
    Feathered, shape-less neutralization (NO rectangles/polygons).
    - Desaturates + slightly dims clothing region via a blurred elliptical mask.
    - Returns (temp_path, torso_box) for light fit scoring.
    - Falls back to no-op if mediapipe is unavailable.
    """
    img = PILImage.open(io.BytesIO(person_bytes)).convert("RGBA")
    w, h = img.size

    # Default torso_box = whole image (safe fallback)
    default_box = (0, 0, w, h)

    if mp_pose is None:
        b = io.BytesIO(); img.save(b, format="PNG")
        return _as_temp_png(b.getvalue()), default_box

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        rgb = np.array(img.convert("RGB"))
        res = pose.process(rgb)

    if not getattr(res, "pose_landmarks", None):
        b = io.BytesIO(); img.save(b, format="PNG")
        return _as_temp_png(b.getvalue()), default_box

    lm = res.pose_landmarks.landmark
    def dn(p): return (int(p.x * w), int(p.y * h))
    shoulders = [dn(lm[11]), dn(lm[12])]
    hips      = [dn(lm[24]), dn(lm[23])]
    xs = [p[0] for p in shoulders + hips]
    ys = [p[1] for p in shoulders + hips]

    # Clothing region estimate
    x_mid = int(sum(xs) / len(xs))
    half  = max(110, int((max(xs) - min(xs)) * 0.9))
    x0, x1 = max(0, x_mid - half), min(w, x_mid + half)
    y0     = max(0, min(ys) - 30)
    y1     = int(h * 0.98)
    torso_box = (x0, y0, x1, y1)

    # Build a soft mask (white=apply adjustment, black=keep original)
    mask = PILImage.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([x0 - 60, y0, x1 + 60, y1], fill=int(255 * alpha))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=70))  # feather edges heavily

    # Desaturate + dim a bit inside mask
    base_rgb = img.convert("RGB")
    sat = ImageEnhance.Color(base_rgb).enhance(0.35)
    bri = ImageEnhance.Brightness(sat).enhance(0.92)
    adjusted = PILImage.merge("RGBA", (*bri.split(), img.split()[3]))

    out = PILImage.composite(adjusted, img, mask)

    b = io.BytesIO(); out.save(b, format="PNG")
    return _as_temp_png(b.getvalue()), torso_box


# ---------- env helpers ----------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1", "true", "t", "yes", "y")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


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
    base_steps: int = _env_int("VTO_BASE_STEPS", 70)
    seed: typing.Optional[int] = (int(os.getenv("VTO_SEED")) if os.getenv("VTO_SEED") else None)
    add_watermark: bool = _env_bool("VTO_ADD_WATERMARK", False)
    model: str = os.getenv("VTO_MODEL", "virtual-try-on-preview-08-04")

    # Vertex by default; allow Google-provided env names too
    use_vertex: bool = _env_bool("VTO_USE_VERTEX", _env_bool("GOOGLE_GENAI_USE_VERTEXAI", True))
    project: typing.Optional[str] = os.getenv("GOOGLE_PROJECT_ID")
    location: str = os.getenv("GOOGLE_LOCATION")


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


# ---------- response decoding helpers ----------
def _decode_base64_field(val: typing.Any) -> typing.Optional[bytes]:
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, str):
        try:
            return base64.b64decode(val)
        except Exception:
            return None
    return None


def _to_png_bytes_from_unknown(obj: typing.Any) -> bytes:
    """
    Robustly convert various image objects to PNG bytes.

    Handles:
      - PIL.Image.Image
      - google.genai.types.Image (via .as_pil_image() / .bytes / .data / .save)
      - dict-like Vertex 'predictions' payloads (bytesBase64Encoded, imageBytes, etc.)
      - plain bytes
    """
    # 1) Already PNG/raw bytes?
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)

    # 2) PIL Image
    if isinstance(obj, PILImage.Image):
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        return buf.getvalue()

    # 3) google.genai.types.Image
    if isinstance(obj, Image):
        as_pil = getattr(obj, "as_pil_image", None)
        if callable(as_pil):
            try:
                pil = as_pil()
                if isinstance(pil, PILImage.Image):
                    return _to_png_bytes_from_unknown(pil)
            except Exception as e:
                log.debug("as_pil_image() failed: %s", e)

        for attr in ("bytes", "image_bytes", "data", "content"):
            val = getattr(obj, attr, None)
            if isinstance(val, (bytes, bytearray)):
                return bytes(val)
            if isinstance(val, str):
                decoded = _decode_base64_field(val)
                if decoded:
                    return decoded

        save_fn = getattr(obj, "save", None)
        if callable(save_fn):
            try:
                buf = io.BytesIO()
                save_fn(buf)  # some SDK versions don’t accept format=
                return buf.getvalue()
            except TypeError:
                buf = io.BytesIO()
                save_fn(buf, "PNG")
                return buf.getvalue()

    # 4) dict-like predictions (Vertex REST shapes)
    if isinstance(obj, dict):
        for k in ("bytesBase64Encoded", "imageBytes", "image", "content"):
            if k in obj:
                b = _decode_base64_field(obj[k])
                if b:
                    return b

    # 5) iterable/sequence of candidates
    if isinstance(obj, (list, tuple)):
        for item in obj:
            try:
                return _to_png_bytes_from_unknown(item)
            except Exception:
                continue

    raise RuntimeError("Unsupported image output type (cannot convert to PNG bytes)")


def _score_fit_against_torso(img_bytes: bytes, ref_box: typing.Tuple[int, int, int, int]) -> float:
    """
    Very light heuristic: more non-background pixels inside the torso rectangle == better fit.
    """
    try:
        pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGBA")
        x0, y0, x1, y1 = ref_box
        crop = pil.crop((x0, y0, x1, y1))
        arr = np.array(crop)
        if arr.shape[2] == 4:
            alpha = arr[..., 3]
            return float((alpha > 10).sum()) / float(alpha.size)
    except Exception:
        pass
    return 0.0


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
    torso_box = None
    if _env_bool("VTO_NEUTRALIZE_TORSO", True):
        person_tmp, torso_box = neutralize_torso_bytes(
            person_bytes,
            alpha=_env_float("VTO_TORSO_ALPHA", 0.70),
            soften=int(_env_int("VTO_TORSO_SOFTEN", 22)),
        )
    else:
        # no-op neutralization path
        pil = PILImage.open(io.BytesIO(person_bytes)).convert("RGBA")
        b = io.BytesIO(); pil.save(b, format="PNG")
        person_tmp = _as_temp_png(b.getvalue())
        torso_box = (0, 0, pil.size[0], pil.size[1])

    garment_tmp = prep_garment_bytes(garment_bytes)

    person_img = Image.from_file(location=person_tmp)
    garment_img = Image.from_file(location=garment_tmp)

    # Recontext config
    try:
        re_cfg = RecontextImageConfig(
            number_of_images=_env_int("VTO_NUM_IMAGES", 4),
            base_steps=cfg.base_steps,
            seed=(cfg.seed or random.randint(1, 10_000_000)),
            add_watermark=cfg.add_watermark,
        )
    except Exception as e:
        log.debug("RecontextImageConfig with extras failed; falling back: %s", e)
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

    # --- Extract candidates ---
    images = (
        getattr(resp, "generated_images", None)
        or getattr(resp, "images", None)
        or getattr(resp, "predictions", None)
        or (resp if isinstance(resp, (list, tuple)) else None)
        or []
    )

    if not images:
        if isinstance(resp, dict) and "predictions" in resp:
            images = resp["predictions"]
        else:
            raise RuntimeError("VTO returned no image")

    # Score & select best candidate
    scored: typing.List[typing.Tuple[float, bytes]] = []
    for cand in images:
        obj = cand.image if hasattr(cand, "image") else cand
        try:
            b = _to_png_bytes_from_unknown(obj)
            score = _score_fit_against_torso(b, torso_box) if torso_box else 0.0
            scored.append((score, b))
        except Exception as e:
            log.debug("candidate decode failed: %s", e)

    if not scored:
        raise RuntimeError("VTO returned images but none were decodable")

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]
