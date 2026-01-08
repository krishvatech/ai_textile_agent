from __future__ import annotations

import hashlib
import json
import logging
import os
from io import BytesIO
from typing import Dict, List, Optional
from uuid import uuid4

from PIL import Image as PILImage

from app.core.virtual_try_on import generate_tryon_candidates
from app.services.product_modelling.models import GenerateCatalogResult
from app.services.product_modelling.prompts import build_prompts
from app.services.product_modelling.vertex_client import (
    caption_reference_style,
    edit_place_product_on_person,
    product_recontext,
)

log = logging.getLogger(__name__)

MAX_IMAGE_BYTES = int(os.getenv("PRODUCT_MODELLING_MAX_BYTES", str(12 * 1024 * 1024)))
VERTEX_TIMEOUT = float(os.getenv("PRODUCT_MODELLING_TIMEOUT", "120"))
MODEL_INDEX_PATH = os.path.join("app", "static", "model_library", "index.json")


def load_model_library_index() -> Dict[str, List[Dict[str, str]]]:
    try:
        with open(MODEL_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        log.warning("Model library index missing or invalid: %s", e)
    return {"women": [], "men": []}


def _find_model_path(model_id: str) -> Optional[str]:
    if not model_id:
        return None
    index = load_model_library_index()
    for group in ("women", "men"):
        for item in index.get(group, []):
            if item.get("id") == model_id:
                return item.get("path")
    return None


def _read_local_model(path: str) -> Optional[bytes]:
    if not path:
        return None
    rel = path.lstrip("/")
    fs_path = os.path.join("app", rel.replace("/", os.sep)) if rel.startswith("static") else os.path.join("app", rel)
    if not os.path.isfile(fs_path):
        return None
    with open(fs_path, "rb") as f:
        return f.read()


def _auto_pick_model_id() -> Optional[str]:
    index = load_model_library_index()
    for group in ("women", "men"):
        items = index.get(group, [])
        if items:
            return items[0].get("id")
    return None


def _save_png(path: str, data: bytes) -> None:
    img = PILImage.open(BytesIO(data)).convert("RGBA")
    img.save(path, format="PNG")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _ensure_unique_images(
    generator,
    prompts: List[str],
    base_args: Dict[str, object],
    desired: int,
) -> List[bytes]:
    out: List[bytes] = []
    seen = set()
    for prompt in prompts:
        if len(out) >= desired:
            break
        images = await generator(prompt=prompt, **base_args)
        for img in images:
            h = _hash_bytes(img)
            if h in seen:
                continue
            seen.add(h)
            out.append(img)
            if len(out) >= desired:
                break
    return out


async def _generate_tryon_unique(
    garment_bytes: bytes,
    person_bytes: bytes,
    desired: int,
) -> List[bytes]:
    out: List[bytes] = []
    seen = set()
    attempts = 0
    while len(out) < desired and attempts < 3:
        attempts += 1
        images = await generate_tryon_candidates(
            person_bytes=person_bytes,
            garment_bytes=garment_bytes,
            num_images=max(desired, 4),
        )
        for img in images:
            h = _hash_bytes(img)
            if h in seen:
                continue
            seen.add(h)
            out.append(img)
            if len(out) >= desired:
                break
    return out


async def generate_catalog(
    *,
    workflow: str,
    product_image_bytes: bytes,
    reference_image_bytes: Optional[bytes],
    model_image_bytes: Optional[bytes],
    predefined_model_id: Optional[str],
    style_preset: Optional[str],
    background: str,
    pose_set: str,
    strict_garment: bool,
    num_images: int,
    tenant_scope: Optional[str],
) -> Dict[str, object]:
    if not product_image_bytes:
        raise ValueError("Product image is required.")
    if len(product_image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Product image exceeds size limit.")

    num_images = max(2, min(int(num_images or 2), 6))
    workflow_in = (workflow or "auto").strip().lower()
    if workflow_in not in {"auto", "accessory", "clothing"}:
        workflow_in = "auto"

    if workflow_in == "auto":
        workflow_used = "clothing" if (model_image_bytes or predefined_model_id) else "accessory"
    else:
        workflow_used = workflow_in

    reference_style = None
    if reference_image_bytes:
        try:
            reference_style = await caption_reference_style(
                reference_image_bytes,
                timeout_s=min(VERTEX_TIMEOUT, 60.0),
            )
        except Exception as e:
            log.warning("Reference captioning failed: %s", e)

    model_id_used = predefined_model_id or ""
    if workflow_used == "clothing" and not model_image_bytes:
        if not model_id_used:
            model_id_used = _auto_pick_model_id() or ""
        if model_id_used:
            model_image_bytes = _read_local_model(_find_model_path(model_id_used) or "")

    subject_hint = None
    if workflow_used == "clothing":
        subject_hint = "model wearing the garment"
    elif workflow_used == "accessory":
        subject_hint = "model wearing the accessory" if model_image_bytes else "isolated product shot, no person"

    prompts = build_prompts(
        workflow=workflow_used,
        background=background,
        pose_set=pose_set,
        strict_garment=bool(strict_garment),
        style_preset=style_preset or None,
        reference_style=reference_style,
        num_images=num_images,
        subject_hint=subject_hint,
    )

    images: List[bytes] = []
    if workflow_used == "clothing":
        if not model_image_bytes:
            raise ValueError("Model image is required for clothing workflow.")
        images = await _generate_tryon_unique(
            garment_bytes=product_image_bytes,
            person_bytes=model_image_bytes,
            desired=num_images,
        )
    elif workflow_used == "accessory":
        if model_image_bytes:
            images = await _ensure_unique_images(
                generator=lambda prompt, **kw: edit_place_product_on_person(
                    prompt=prompt, timeout_s=VERTEX_TIMEOUT, **kw
                ),
                prompts=prompts,
                base_args={
                    "person_image_bytes": model_image_bytes,
                    "product_image_bytes": product_image_bytes,
                    "num_images": 1,
                },
                desired=num_images,
            )
        else:
            images = await _ensure_unique_images(
                generator=lambda prompt, **kw: product_recontext(
                    prompt=prompt, timeout_s=VERTEX_TIMEOUT, **kw
                ),
                prompts=prompts,
                base_args={
                    "product_image_bytes": product_image_bytes,
                    "num_images": 1,
                },
                desired=num_images,
            )
    else:
        images = await _ensure_unique_images(
            generator=lambda prompt, **kw: product_recontext(
                prompt=prompt, timeout_s=VERTEX_TIMEOUT, **kw
            ),
            prompts=prompts,
            base_args={
                "product_image_bytes": product_image_bytes,
                "num_images": 1,
            },
            desired=num_images,
        )

    if len(images) < num_images:
        raise RuntimeError("Vertex AI did not return enough distinct images.")

    job_id = uuid4().hex
    scope = tenant_scope or "admin"
    if scope != "admin" and not scope.startswith("tenant_"):
        scope = f"tenant_{scope}"
    base_dir = os.path.join("app", "static", "generated", "modelling", scope, job_id)
    os.makedirs(base_dir, exist_ok=True)

    _save_png(os.path.join(base_dir, "input_product.png"), product_image_bytes)
    if reference_image_bytes:
        _save_png(os.path.join(base_dir, "input_reference.png"), reference_image_bytes)
    if model_image_bytes:
        _save_png(os.path.join(base_dir, "input_model.png"), model_image_bytes)

    public_prefix = f"/static/generated/modelling/{scope}/{job_id}"
    urls: List[str] = []
    for idx, img in enumerate(images, start=1):
        out_name = f"out_{idx:02d}.png"
        _save_png(os.path.join(base_dir, out_name), img)
        urls.append(f"{public_prefix}/{out_name}")

    result = GenerateCatalogResult(
        job_id=job_id,
        images=urls,
        meta={
            "workflow": workflow_used,
            "prompts": prompts,
            "background": background,
            "pose_set": pose_set,
            "strict_garment": bool(strict_garment),
            "style_preset": style_preset or "",
            "num_images": num_images,
            "predefined_model_id": model_id_used or "",
            "tenant_scope": scope,
            "reference_style": reference_style or {},
            "models": {
                "product_recontext": os.getenv("PRODUCT_RECONTEXT_MODEL", "imagen-product-recontext-preview-06-30"),
                "edit_model": os.getenv("IMAGEN_EDIT_MODEL", "imagen-3.0-capability-001"),
                "vto_model": os.getenv("VTO_MODEL", "virtual-try-on-preview-08-04"),
            },
        },
    )
    return result.model_dump()
