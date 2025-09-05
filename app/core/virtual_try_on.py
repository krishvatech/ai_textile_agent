# app/services/virtual_try_on.py
import os, io, tempfile, random, typing, asyncio
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


# ---------- helpers ----------
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


@dataclass
class VTOConfig:
    base_steps: int = 60
    seed: int | None = None
    add_watermark: bool = False
    model: str = "virtual-try-on-preview-08-04"  # Google GenAI Recontext image model
    use_vertex: bool = bool(int(os.getenv("VTO_USE_VERTEX", "0")))
    project: str | None = os.getenv("GOOGLE_PROJECT_ID")
    location: str | None = os.getenv("GOOGLE_LOCATION", "us-central1")

def _make_client(cfg: VTOConfig) -> genai.Client:
    if cfg.use_vertex:
        # Vertex (only if you explicitly enable it)
        return genai.Client(
            vertexai=True,
            project=cfg.project,
            location=cfg.location,
            http_options=HttpOptions(api_version="v1"),
        )
    # Pure Google GenAI (no Vertex)
    return genai.Client(http_options=HttpOptions(api_version="v1"))

def _png_to_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


async def generate_vto_image(
    person_bytes: bytes,
    garment_bytes: bytes,
    cfg: VTOConfig | None = None,
) -> bytes:
    """
    Main entry: returns PNG bytes of the try-on result.
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
    # Fallback if type unknown but has .save()
    if hasattr(out, "save"):
        buf = io.BytesIO(); out.save(buf, format="PNG"); return buf.getvalue()
    raise RuntimeError("Unsupported VTO output type")
