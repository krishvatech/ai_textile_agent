# # app/services/virtual_try_on.py
# import os, io, tempfile, random, asyncio, logging, json
# from uuid import uuid4
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional
# from PIL import Image as PILImage, ImageDraw
# import numpy as np

# # Google GenAI (python -m pip install google-genai)
# from google import genai
# from google.genai.types import (
#     HttpOptions, RecontextImageSource, ProductImage, Image, RecontextImageConfig
# )

# # optional deps
# try:
#     from rembg import remove as rembg_remove
# except Exception:
#     rembg_remove = None

# try:
#     import mediapipe as mp
#     mp_pose = mp.solutions.pose
# except Exception:
#     mp_pose = None


# # ========== CREDENTIALS RESOLUTION (service account) ==========
# def _candidate_cred_paths() -> list[Path]:
#     """Preferred → fallback search order for credentials.json."""
#     env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
#     return [
#         Path(env_path).expanduser() if env_path else None,
#         Path.cwd() / "credentials.json",
#         Path(__file__).resolve().parent.parent / "credentials.json",  # project root if this file is in app/services/
#         Path(__file__).resolve().parent / "credentials.json",
#     ]

# def _resolve_credentials_path() -> Path:
#     for p in _candidate_cred_paths():
#         if p and p.is_file():
#             return p
#     raise FileNotFoundError(
#         "Google credentials.json not found. "
#         "Set GOOGLE_APPLICATION_CREDENTIALS or place credentials.json at project root."
#     )

# def _ensure_google_adc() -> tuple[str, str]:
#     """
#     Ensures GOOGLE_APPLICATION_CREDENTIALS points to a readable file.
#     Returns (creds_path, project_id).
#     """
#     cred_path = _resolve_credentials_path()
#     if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
#         logging.info("[VTO][CREDS] Set GOOGLE_APPLICATION_CREDENTIALS=%s", cred_path)

#     # Read project_id from the file if GOOGLE_PROJECT_ID is not set
#     project_id = os.getenv("GOOGLE_PROJECT_ID")
#     if not project_id:
#         try:
#             with open(cred_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#             project_id = (data or {}).get("project_id")
#             if project_id:
#                 os.environ.setdefault("GOOGLE_PROJECT_ID", project_id)
#                 os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)  # helpful for some libs
#                 logging.info("[VTO][CREDS] Derived GOOGLE_PROJECT_ID from credentials.json: %s", project_id)
#             else:
#                 raise ValueError("project_id missing in credentials.json")
#         except Exception as e:
#             raise RuntimeError(
#                 "Could not determine GOOGLE_PROJECT_ID. "
#                 "Set it in the environment or include 'project_id' in credentials.json."
#             ) from e

#     return str(cred_path), project_id
# # =============================================================


# # ---------- image helpers ----------
# def _as_temp_png(data: bytes) -> str:
#     p = os.path.join(tempfile.gettempdir(), f"vto_{uuid4().hex}.png")
#     with open(p, "wb") as f:
#         f.write(data)
#     return p

# def _tight_bbox_from_rgba(img_rgba: PILImage.Image):
#     arr = np.array(img_rgba)
#     if arr.shape[2] == 4:
#         alpha = arr[..., 3]
#         ys, xs = np.where(alpha > 0)
#         if len(xs) and len(ys):
#             return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
#     return None

# def prep_garment_bytes(garment_bytes: bytes) -> str:
#     """BG removal + tight crop from raw bytes → temp PNG path."""
#     img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")
#     if rembg_remove:
#         try:
#             garment_bytes = rembg_remove(garment_bytes)
#             img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")
#         except Exception:
#             pass
#     box = _tight_bbox_from_rgba(img)
#     if box:
#         img = img.crop(box)
#     b2 = io.BytesIO(); img.save(b2, format="PNG")
#     return _as_temp_png(b2.getvalue())

# def neutralize_torso_bytes(person_bytes: bytes, alpha=0.7, soften=22) -> str:
#     """Softly mute torso region so the model replaces it cleanly."""
#     img = PILImage.open(io.BytesIO(person_bytes)).convert("RGBA")
#     w, h = img.size
#     if mp_pose is None:
#         b = io.BytesIO(); img.save(b, format="PNG")
#         return _as_temp_png(b.getvalue())

#     with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
#         rgb = np.array(img.convert("RGB"))
#         res = pose.process(rgb)

#     if not getattr(res, "pose_landmarks", None):
#         b = io.BytesIO(); img.save(b, format="PNG")
#         return _as_temp_png(b.getvalue())

#     lm = res.pose_landmarks.landmark
#     def dn(p): return (int(p.x * w), int(p.y * h))
#     pts = [dn(lm[11]), dn(lm[12]), dn(lm[24]), dn(lm[23])]
#     xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
#     x0, x1 = max(0, min(xs) - 20), min(w, max(xs) + 20)
#     y0, y1 = max(0, min(ys) - 30), min(h, max(ys) + 40)

#     overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)
#     draw.rounded_rectangle([x0, y0, x1, y1], radius=soften, fill=(180, 180, 180, int(255 * alpha)))
#     out = PILImage.alpha_composite(img, overlay)
#     b = io.BytesIO(); out.save(b, format="PNG")
#     return _as_temp_png(b.getvalue())


# # ---------- config & client ----------
# @dataclass
# class VTOConfig:
#     base_steps: int = 60
#     seed: Optional[int] = None
#     add_watermark: bool = False
#     model: str = "virtual-try-on-preview-08-04"  # Google GenAI Recontext image model
#     # Use service-account / Vertex when true (recommended for server)
#     use_vertex: bool = bool(int(os.getenv("VTO_USE_VERTEX", "1")))
#     project: Optional[str] = os.getenv("GOOGLE_PROJECT_ID")  # may be filled from credentials.json
#     location: Optional[str] = os.getenv("GOOGLE_LOCATION", "us-central1")

# def _make_client(cfg: VTOConfig) -> genai.Client:
#     """
#     - If use_vertex=True → use service-account via ADC (GOOGLE_APPLICATION_CREDENTIALS)
#     - Else → expects a Google AI Studio API key (not covered here)
#     """
#     if cfg.use_vertex:
#         cred_path, project_id = _ensure_google_adc()
#         project = cfg.project or project_id
#         location = cfg.location or "us-central1"
#         logging.info("[VTO][CLIENT] Vertex mode | project=%s | location=%s | creds=%s",
#                      project, location, cred_path)
#         return genai.Client(
#             vertexai=True,
#             project=project,
#             location=location,
#             http_options=HttpOptions(api_version="v1"),
#         )

#     # Non-Vertex usage would require an API key:
#     # os.environ["GOOGLE_API_KEY"] must be set (not service-account)
#     logging.info("[VTO][CLIENT] Non-Vertex mode (API key).")
#     return genai.Client(http_options=HttpOptions(api_version="v1"))


# # ---------- VTO core ----------
# async def generate_vto_image(
#     person_bytes: bytes,
#     garment_bytes: bytes,
#     cfg: VTOConfig | None = None,
# ) -> bytes:
#     cfg = cfg or VTOConfig()
#     client = _make_client(cfg)

#     # Prep inputs (unchanged)
#     person_tmp  = neutralize_torso_bytes(person_bytes)
#     garment_tmp = prep_garment_bytes(garment_bytes)
#     logging.info("[VTO][INPUTS] person_tmp=%s | garment_tmp=%s", person_tmp, garment_tmp)

#     person_img  = Image.from_file(location=person_tmp)
#     garment_img = Image.from_file(location=garment_tmp)

#     # Seed & steps (unchanged)
#     try:
#         re_cfg = RecontextImageConfig(
#             number_of_images=1,
#             base_steps=cfg.base_steps,
#             seed=(cfg.seed or random.randint(1, 10_000_000)),
#             add_watermark=cfg.add_watermark,
#         )
#     except Exception:
#         re_cfg = RecontextImageConfig(number_of_images=1)

#     # ✅ FIXED: call with keyword args; provide a "source=" fallback
#     logging.info("[VTO][CALL] model=%s | steps=%s", cfg.model, getattr(re_cfg, "base_steps", None))

#     def _call_recontext():
#         try:
#             # Most current builds expect `image=` for the source payload.
#             return client.models.recontext_image(
#                 model=cfg.model,
#                 image=RecontextImageSource(
#                     person_image=person_img,
#                     product_images=[ProductImage(product_image=garment_img)],
#                 ),
#                 config=re_cfg,
#             )
#         except TypeError:
#             # Some earlier builds used `source=`.
#             logging.info("[VTO][CALL] Retrying recontext_image with source=")
#             return client.models.recontext_image(
#                 model=cfg.model,
#                 source=RecontextImageSource(
#                     person_image=person_img,
#                     product_images=[ProductImage(product_image=garment_img)],
#                 ),
#                 config=re_cfg,
#             )

#     resp = await asyncio.to_thread(_call_recontext)

#     # Handle output (unchanged)
#     imgs = getattr(resp, "generated_images", []) or []
#     if not imgs:
#         raise RuntimeError("VTO returned no image")

#     out = imgs[0].image
#     if isinstance(out, PILImage.Image):
#         buf = io.BytesIO(); out.save(buf, format="PNG")
#         logging.info("[VTO][OK] Generated PIL image (PNG %d bytes)", buf.getbuffer().nbytes)
#         return buf.getvalue()
#     if isinstance(out, bytes):
#         logging.info("[VTO][OK] Generated raw bytes (len=%d)", len(out))
#         return out
#     if hasattr(out, "save"):
#         buf = io.BytesIO(); out.save(buf, format="PNG")
#         logging.info("[VTO][OK] Generated image via .save() (PNG %d bytes)", buf.getbuffer().nbytes)
#         return buf.getvalue()

#     raise RuntimeError("Unsupported VTO output type")

# app/core/virtual_try_on.py
import os, io, tempfile, random, typing, asyncio, json, logging
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


# -------------------- credentials / client --------------------
def _find_credentials_file() -> str | None:
    # Prefer explicit env; otherwise ./credentials.json (your server screenshot)
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    cand = os.path.join(os.getcwd(), "credentials.json")
    return cand if os.path.isfile(cand) else None

def _ensure_vertex_env():
    cred_path = _find_credentials_file()
    if cred_path and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

    project = os.getenv("GOOGLE_PROJECT_ID")
    if (not project) and cred_path and os.path.isfile(cred_path):
        try:
            with open(cred_path, "r") as f:
                data = json.load(f)
            project = data.get("project_id") or project
            if project:
                os.environ["GOOGLE_PROJECT_ID"] = project
                logging.info("[VTO][CREDS] Derived GOOGLE_PROJECT_ID from credentials.json: %s", project)
        except Exception:
            pass

@dataclass
class VTOConfig:
    base_steps: int = 60
    seed: int | None = None
    add_watermark: bool = False
    model: str = "virtual-try-on-preview-08-04"
    use_vertex: bool = True
    project: str | None = os.getenv("GOOGLE_PROJECT_ID")
    location: str | None = os.getenv("GOOGLE_LOCATION", "us-central1")
    src_input_padding: float = 0.15

def _make_client(cfg: VTOConfig) -> genai.Client:
    _ensure_vertex_env()
    if cfg.use_vertex:
        project = cfg.project or os.getenv("GOOGLE_PROJECT_ID")
        location = cfg.location or "us-central1"
        logging.info(
            "[VTO][CLIENT] Vertex mode | project=%s | location=%s | creds=%s",
            project, location, os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )
    return genai.Client(http_options=HttpOptions(api_version="v1"))


# -------------------- main entry --------------------
async def generate_vto_image(
    person_bytes: bytes,
    garment_bytes: bytes,
    cfg: VTOConfig | None = None,
) -> bytes:
    """
    Returns PNG bytes of the try-on result.

    Correct request for Vertex `recontext_image`:
      - Build RecontextImageSource with **src_input** (person) and **src_input_padding**
      - Include garment as ProductImage(product_image=...)
      - Call recontext_image(model=..., source=..., config=...)
    """
    cfg = cfg or VTOConfig()
    client = _make_client(cfg)

    # Preprocess inputs
    person_tmp  = neutralize_torso_bytes(person_bytes)
    garment_tmp = prep_garment_bytes(garment_bytes)
    logging.info("[VTO][INPUTS] person_tmp=%s | garment_tmp=%s", person_tmp, garment_tmp)

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

    logging.info("[VTO][CALL] model=%s | steps=%s", cfg.model, cfg.base_steps)

    def _call_recontext():
        source = RecontextImageSource(
            # *** THIS IS THE FIX ***
            src_input=person_img,
            src_input_padding=cfg.src_input_padding,
            product_images=[ProductImage(product_image=garment_img)],
        )
        return client.models.recontext_image(
            model=cfg.model,
            source=source,
            config=re_cfg,
        )

    # Do the call on a worker thread
    resp = await asyncio.to_thread(_call_recontext)

    # Parse response
    imgs = getattr(resp, "generated_images", []) or []
    if not imgs:
        raise RuntimeError("VTO returned no image")

    out = imgs[0].image
    # Normalize to PNG bytes
    if isinstance(out, PILImage.Image):
        buf = io.BytesIO(); out.save(buf, format="PNG")
        logging.info("[VTO][SUCCESS] image generated (PIL)")
        return buf.getvalue()
    if isinstance(out, bytes):
        logging.info("[VTO][SUCCESS] image generated (bytes)")
        return out
    if hasattr(out, "save"):
        buf = io.BytesIO(); out.save(buf, format="PNG")
        logging.info("[VTO][SUCCESS] image generated (saveable)")
        return buf.getvalue()

    raise RuntimeError("Unsupported VTO output type")
