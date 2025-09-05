# app/services/virtual_try_on.py
"""
Environment-only configuration for Google VTO.

Required env vars (no hardcoded paths):
- GOOGLE_APPLICATION_CREDENTIALS   -> path to service-account JSON (e.g. D:\ai_textile_agent\credentials.json)
- GOOGLE_PROJECT_ID                -> optional; if missing we'll read project_id from the JSON
- GOOGLE_LOCATION                  -> optional; default 'us-central1'
- VTO_USE_VERTEX                   -> default '1' (true). If '0', requires GOOGLE_API_KEY for non-Vertex.
- VTO_MODEL                        -> optional; default 'virtual-try-on-preview-08-04'
- VTO_BASE_STEPS                   -> optional; default '60'
- VTO_SEED                         -> optional; integer; unset = random
- VTO_ADD_WATERMARK                -> optional; '1'/'true' enables watermark

Notes:
- With VTO_USE_VERTEX=1, auth uses Application Default Credentials from GOOGLE_APPLICATION_CREDENTIALS.
- With VTO_USE_VERTEX=0, set GOOGLE_API_KEY for Google AI (non-Vertex) endpoints.
"""

import os, io, tempfile, random, typing, asyncio, json
from uuid import uuid4
from dataclasses import dataclass
from PIL import Image as PILImage, ImageDraw
import numpy as np

# Google GenAI
from google import genai
from google.genai.types import (
    HttpOptions, RecontextImageSource, ProductImage, Image, RecontextImageConfig
)

# optional deps
try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None


# -------------------- helpers --------------------
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
            return int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
    return None

def _load_image_bytes_to_png_path(data: bytes) -> str:
    try:
        img = PILImage.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        # fallback—let PIL try again as RGB
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return _as_temp_png(buf.getvalue())
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return _as_temp_png(buf.getvalue())

def prep_garment_bytes(garment_bytes: bytes) -> str:
    """BG removal + tight crop from raw bytes → temp PNG path."""
    img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")
    if rembg_remove:
        try:
            garment_bytes = rembg_remove(garment_bytes)
            img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")
        except Exception:
            pass
    box = _tight_bbox_from_rgba(img)
    if box:
        img = img.crop(box)
    b2 = io.BytesIO(); img.save(b2, format="PNG")
    return _as_temp_png(b2.getvalue())

def neutralize_torso_bytes(person_bytes: bytes, alpha=0.7, soften=22) -> str:
    """Softly mute torso region so the model replaces it cleanly."""
    img = PILImage.open(io.BytesIO(person_bytes)).convert("RGBA")
    w, h = img.size
    if mp_pose is None:
        b = io.BytesIO(); img.save(b, format="PNG")
        return _as_temp_png(b.getvalue())

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        rgb = np.array(img.convert("RGB"))
        res = pose.process(rgb)

    if not getattr(res, "pose_landmarks", None):
        b = io.BytesIO(); img.save(b, format="PNG")
        return _as_temp_png(b.getvalue())

    lm = res.pose_landmarks.landmark
    def dn(p): return (int(p.x*w), int(p.y*h))
    pts = [dn(lm[11]), dn(lm[12]), dn(lm[24]), dn(lm[23])]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = max(0, min(xs)-20), min(w, max(xs)+20)
    y0, y1 = max(0, min(ys)-30), min(h, max(ys)+40)

    overlay = PILImage.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle([x0,y0,x1,y1], radius=soften, fill=(180,180,180,int(255*alpha)))
    out = PILImage.alpha_composite(img, overlay)
    b = io.BytesIO(); out.save(b, format="PNG")
    return _as_temp_png(b.getvalue())


# -------------------- env utils --------------------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1", "true", "t", "yes", "y")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _project_from_adc() -> str | None:
    """
    Read project_id from the service-account JSON pointed to by GOOGLE_APPLICATION_CREDENTIALS.
    Returns None if unavailable.
    """
    adc = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not adc or not os.path.isfile(adc):
        return None
    try:
        with open(adc, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Standard SA key has project_id field
        return data.get("project_id")
    except Exception:
        return None


# -------------------- config --------------------
@dataclass
class VTOConfig:
    base_steps: int = _env_int("VTO_BASE_STEPS", 60)
    seed: int | None = (int(os.getenv("VTO_SEED")) if os.getenv("VTO_SEED") else None)
    add_watermark: bool = _env_bool("VTO_ADD_WATERMARK", False)
    model: str = os.getenv("VTO_MODEL", "virtual-try-on-preview-08-04")
    use_vertex: bool = _env_bool("VTO_USE_VERTEX", True)  # default to Vertex (works with service-account)
    project: str | None = (os.getenv("GOOGLE_PROJECT_ID") or _project_from_adc())
    location: str | None = os.getenv("GOOGLE_LOCATION", "us-central1")


def _validate_adc_or_die():
    adc = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not adc:
        raise EnvironmentError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. "
            "Point it to your service-account JSON file."
        )
    if not os.path.isfile(adc):
        raise EnvironmentError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {adc}"
        )


def _make_client(cfg: VTOConfig) -> genai.Client:
    """
    Returns a configured genai.Client using env-only auth:
    - Vertex path (default): ADC from GOOGLE_APPLICATION_CREDENTIALS + project/location.
    - Non-Vertex path: requires GOOGLE_API_KEY in env.
    """
    if cfg.use_vertex:
        _validate_adc_or_die()
        if not cfg.project:
            raise EnvironmentError(
                "GOOGLE_PROJECT_ID is not set and project_id could not be read from "
                "the service-account JSON. Please set GOOGLE_PROJECT_ID."
            )
        return genai.Client(
            vertexai=True,
            project=cfg.project,
            location=cfg.location,
            http_options=HttpOptions(api_version="v1"),
        )

    # Non-Vertex (Google AI Studio endpoints) requires an API key from env.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "VTO_USE_VERTEX=0 requires GOOGLE_API_KEY in the environment."
        )
    return genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))


# -------------------- main API --------------------
async def generate_vto_image(
    person_bytes: bytes,
    garment_bytes: bytes,
    cfg: VTOConfig | None = None,
) -> bytes:
    """
    Main entry: returns PNG bytes of the try-on result.
    Fully env-driven configuration. No hardcoded paths/keys in code.
    """
    cfg = cfg or VTOConfig()
    client = _make_client(cfg)

    # prep inputs
    person_tmp  = neutralize_torso_bytes(person_bytes)
    garment_tmp = prep_garment_bytes(garment_bytes)

    person_img  = Image.from_file(location=person_tmp)
    garment_img = Image.from_file(location=garment_tmp)

    # seed & steps
    try:
        re_cfg = RecontextImageConfig(
            number_of_images=1,
            base_steps=cfg.base_steps,
            seed=(cfg.seed or random.randint(1, 10_000_000)),
            add_watermark=cfg.add_watermark,
        )
    except Exception:
        re_cfg = RecontextImageConfig(number_of_images=1)

    # call VTO
    resp = await asyncio.to_thread(
        client.models.recontext_image,
        cfg.model,
        RecontextImageSource(
            person_image=person_img,
            product_images=[ProductImage(product_image=garment_img)],
        ),
        re_cfg,
    )

    imgs = getattr(resp, "generated_images", []) or []
    if not imgs:
        raise RuntimeError("VTO returned no image")

    # Always return first
    out = imgs[0].image
    # Normalize to PNG bytes
    if isinstance(out, PILImage.Image):
        buf = io.BytesIO(); out.save(buf, format="PNG"); return buf.getvalue()
    if isinstance(out, bytes):
        return out
    if hasattr(out, "save"):
        buf = io.BytesIO(); out.save(buf, format="PNG"); return buf.getvalue()
    raise RuntimeError("Unsupported VTO output type")
