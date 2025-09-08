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
from PIL import Image as PILImage, ImageDraw

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
    import cv2
except Exception:
    cv2 = None

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
    BG remove (if rembg present) + tight crop + safe max-side → temp PNG path.
    """
    img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")

    # optional background remove
    if rembg_remove:
        try:
            garment_bytes = rembg_remove(garment_bytes)
            img = PILImage.open(io.BytesIO(garment_bytes)).convert("RGBA")
        except Exception as e:
            log.debug("rembg failed (non-fatal): %s", e)

    # tight bbox from alpha
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



def neutralize_torso_bytes(person_bytes: bytes, alpha=0.7, soften=22) -> typing.Union[str, typing.Tuple[str, typing.Tuple[int,int,int,int]]]:
    """
    Softly mute torso so the model replaces garment cleanly.
    Falls back to no-op if mediapipe is unavailable or landmarks missing.
    """
    img = PILImage.open(io.BytesIO(person_bytes)).convert("RGBA")
    w, h = img.size
    if mp_pose is None:
        b = io.BytesIO(); img.save(b, format="PNG"); return _as_temp_png(b.getvalue()), (0,0,img.size[0], img.size[1])

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        rgb = np.array(img.convert("RGB"))
        res = pose.process(rgb)

    if not getattr(res, "pose_landmarks", None):
        b = io.BytesIO(); img.save(b, format="PNG"); return _as_temp_png(b.getvalue()), (0,0,img.size[0], img.size[1])

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
    return _as_temp_png(b.getvalue()), (x0, y0, x1, y1)

def _inpaint_objects_near_hands(person_bytes: bytes,
                                dilate_px: int = 20,
                                strength: int = 35) -> bytes:
    """
    Detect hands (MediaPipe), expand palm/hand regions, and inpaint inside them.
    Removes objects being held without relying on color.
    """
    if mp_pose is None or cv2 is None:
        # if mediapipe or opencv missing, return original
        return person_bytes

    img_rgba = PILImage.open(io.BytesIO(person_bytes)).convert("RGBA")
    w, h = img_rgba.size
    # work in BGR for OpenCV
    bgr = cv2.cvtColor(np.array(img_rgba.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Use MediaPipe Hands (lightweight) if available; else fallback to Pose wrists
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
    except Exception:
        mp_hands = None

    mask = np.zeros((h, w), dtype=np.uint8)

    if mp_hands is not None:
        with mp_hands.Hands(static_image_mode=True,
                            max_num_hands=2,
                            model_complexity=0) as hands:
            res = hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                pts = []
                for lm in hand.landmark:
                    pts.append((int(lm.x * w), int(lm.y * h)))
                # convex hull around all hand keypoints
                hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                cv2.fillConvexPoly(mask, hull, 255)
    else:
        # Fallback: use wrists+elbows from Pose to approximate a small circular area at wrists
        with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if getattr(res, "pose_landmarks", None):
            lm = res.pose_landmarks.landmark
            for idx in (15, 16):  # left/right wrist
                x = int(lm[idx].x * w); y = int(lm[idx].y * h)
                cv2.circle(mask, (x, y), 40, 255, -1)

    if mask.max() == 0:
        # nothing detected
        return person_bytes

    # dilate to cover phone/objects around the palm
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, iterations=1)

    # inpaint (Telea) on the RGB image
    inpainted = cv2.inpaint(bgr, mask, strength, cv2.INPAINT_TELEA)

    # keep original alpha
    rgba = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    out = PILImage.fromarray(rgba).convert("RGBA")
    out.putalpha(img_rgba.split()[-1])

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()

def _safe_pose_landmarks(img_rgba: PILImage.Image):
    if mp_pose is None:
        return None, img_rgba.size
    w, h = img_rgba.size
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        rgb = np.array(img_rgba.convert("RGB"))
        res = pose.process(rgb)
    return getattr(res, "pose_landmarks", None), (w, h)

def _draw_arm_mask(mask_draw: ImageDraw.ImageDraw, lm, w, h, width_px: int):
    def pt(idx): 
        p = lm.landmark[idx]
        return (int(p.x * w), int(p.y * h))
    # left arm: 11-13-15, right arm: 12-14-16
    for a,b,c in ((11,13,15), (12,14,16)):
        p1, p2, p3 = pt(a), pt(b), pt(c)
        mask_draw.line([p1, p2], width=width_px, fill=255)
        mask_draw.line([p2, p3], width=width_px, fill=255)

def _preserve_face_and_arms(person_bytes: bytes, gen_bytes: bytes,
                            keep_face=True, keep_arms=True) -> bytes:
    """Paste original face/arms back on top of generated try-on."""
    orig = PILImage.open(io.BytesIO(person_bytes)).convert("RGBA")
    gen  = PILImage.open(io.BytesIO(gen_bytes)).convert("RGBA").resize(orig.size, PILImage.LANCZOS)

    lm, (w, h) = _safe_pose_landmarks(orig)
    if lm is None or (not keep_face and not keep_arms):
        # nothing to preserve
        buf = io.BytesIO(); gen.save(buf, format="PNG"); return buf.getvalue()

    mask = PILImage.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Arms mask (thick lines shoulder→elbow→wrist)
    if keep_arms:
        # width ~1.5% of image diagonal; min 16px, max 48px
        diag = (w**2 + h**2) ** 0.5
        width_px = max(16, min(48, int(diag * 0.015)))
        _draw_arm_mask(draw, lm, w, h, width_px)

    # Face mask (disc around nose; radius from shoulder span)
    if keep_face:
        # shoulders 11,12
        sx = (lm.landmark[11].x + lm.landmark[12].x) * 0.5 * w
        sy = (lm.landmark[11].y + lm.landmark[12].y) * 0.5 * h
        nose = lm.landmark[0]
        nx, ny = int(nose.x * w), int(nose.y * h)
        r = max(24, int(abs(nx - sx) * 0.7))  # approx face radius
        draw.ellipse([nx - r, ny - r, nx + r, ny + r], fill=255)

    # Paste original pixels where mask=1
    out = gen.copy()
    out.paste(orig, (0, 0), mask)
    buf = io.BytesIO(); out.save(buf, format="PNG")
    return buf.getvalue()


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
    project: typing.Optional[str] = (
        os.getenv("GOOGLE_PROJECT_ID")
    )
    location: str = (
        os.getenv("GOOGLE_LOCATION")
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

    # 3) google.genai.types.Image (the class we imported as Image)
    if isinstance(obj, Image):
        # Try a direct .as_pil_image()
        as_pil = getattr(obj, "as_pil_image", None)
        if callable(as_pil):
            try:
                pil = as_pil()
                if isinstance(pil, PILImage.Image):
                    return _to_png_bytes_from_unknown(pil)
            except Exception as e:
                log.debug("as_pil_image() failed: %s", e)

        # Known byte-like fields
        for attr in ("bytes", "image_bytes", "data", "content"):
            val = getattr(obj, attr, None)
            if isinstance(val, (bytes, bytearray)):
                return bytes(val)
            if isinstance(val, str):
                decoded = _decode_base64_field(val)
                if decoded:
                    return decoded

        # Last resort: an object-level .save that takes only a file-like param
        save_fn = getattr(obj, "save", None)
        if callable(save_fn):
            try:
                buf = io.BytesIO()
                # Do NOT pass 'format' — some SDK versions don't accept it
                save_fn(buf)
                return buf.getvalue()
            except TypeError:
                # try with a positional "PNG" just in case
                try:
                    buf = io.BytesIO()
                    save_fn(buf, "PNG")
                    return buf.getvalue()
                except Exception as e:
                    log.debug("obj.save fallback failed: %s", e)

    # 4) dict-like predictions (Vertex REST shapes)
    if isinstance(obj, dict):
        # common keys: bytesBase64Encoded, imageBytes, image, content
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

def _score_fit_against_torso(img_bytes: bytes, ref_box: typing.Tuple[int,int,int,int]) -> float:
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
    if _env_bool("VTO_REMOVE_HAND_OBJECTS", True):
        person_bytes = _inpaint_objects_near_hands(
            person_bytes,
            dilate_px=_env_int("VTO_HAND_DILATION", 22),
            strength=_env_int("VTO_INPAINT_STRENGTH", 35),
        )

    # Continue with your existing flow:
    torso_box = None
    if _env_bool("VTO_NEUTRALIZE_TORSO", True):
        person_tmp, torso_box = neutralize_torso_bytes(
            person_bytes,
            alpha=_env_float("VTO_TORSO_ALPHA", 0.70),
            soften=int(_env_int("VTO_TORSO_SOFTEN", 22)),
        )
    else:
        b = io.BytesIO(); PILImage.open(io.BytesIO(person_bytes)).convert("RGBA").save(b, format="PNG")
        person_tmp = _as_temp_png(b.getvalue())
        pil = PILImage.open(io.BytesIO(person_bytes)).convert("RGBA")
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

    # --- Extract first image-like artifact from response ---
    # Common shapes:
    #   - resp.generated_images -> [GeneratedImage(image=Image(...)), ...]
    #   - resp.images -> [Image | PIL.Image | bytes, ...]
    #   - resp.predictions -> [ { bytesBase64Encoded: ... }, ... ]
    images = (
        getattr(resp, "generated_images", None)
        or getattr(resp, "images", None)
        or getattr(resp, "predictions", None)
        or (resp if isinstance(resp, (list, tuple)) else None)
        or []
    )

    if not images:
        # dict response with predictions
        if isinstance(resp, dict) and "predictions" in resp:
            images = resp["predictions"]
        else:
            raise RuntimeError("VTO returned no image")

    # Prefer nested .image field if present (GeneratedImage)
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
    best_png = scored[0][1]

    # --- NEW: restore original arms/face on top to avoid "third hand"
    if _env_bool("VTO_PRESERVE_ARMS", True) or _env_bool("VTO_PRESERVE_FACE", True):
        best_png = _preserve_face_and_arms(
            person_bytes,
            best_png,
            keep_face=_env_bool("VTO_PRESERVE_FACE", True),
            keep_arms=_env_bool("VTO_PRESERVE_ARMS", True),
        )

    return best_png