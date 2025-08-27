from __future__ import annotations
import os, io
from typing import Any, Dict, List, Optional
from PIL import Image
import torch, open_clip
from pinecone import Pinecone

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX", "ai-textile-agent")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
MODEL_NAME_IMAGE   = "ViT-B-32-quickgelu"

if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY")

_pc_index = None
_clip_model = None
_clip_preprocess = None
_device = "cpu"

def _ensure_backends():
    global _pc_index, _clip_model, _clip_preprocess
    if _pc_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pc_index = pc.Index(PINECONE_INDEX)
    if _clip_model is None or _clip_preprocess is None:
        try:
            torch.set_num_threads(min(4, (os.cpu_count() or 4)))
        except Exception:
            pass
        m, _, pp = open_clip.create_model_and_transforms(MODEL_NAME_IMAGE, pretrained="openai")
        m.eval().to(_device)
        _clip_model, _clip_preprocess = m, pp
    return _pc_index, _clip_model, _clip_preprocess

def embed_image_bytes(image_bytes: bytes) -> List[float]:
    _, model, preprocess = _ensure_backends()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.inference_mode():
        x = preprocess(img).unsqueeze(0).to(_device)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype("float32").tolist()

def _build_filter(tenant_id: Optional[int], modality: str = "image") -> Optional[Dict[str, Any]]:
    f: Dict[str, Any] = {}
    if tenant_id is not None:
        f["tenant_id"] = {"$eq": int(tenant_id)}
    if modality in ("image", "text"):
        f["modality"] = {"$eq": modality}
    return f or None

def visual_search_bytes_sync(image_bytes: bytes, *, tenant_id: Optional[int] = None,
                             top_k: int = 12, modality: str = "image") -> List[Dict[str, Any]]:
    index, _, _ = _ensure_backends()
    vec = embed_image_bytes(image_bytes)
    res = index.query(
        vector=vec, top_k=int(top_k), include_metadata=True,
        namespace=PINECONE_NAMESPACE, filter=_build_filter(tenant_id, modality)
    )
    return (res or {}).get("matches") or []

def format_matches_for_whatsapp(matches: List[Dict[str, Any]], limit: int = 5) -> str:
    if not matches:
        return "No visually similar items found."
    lines = []
    for i, m in enumerate(matches[:limit], 1):
        md = m.get("metadata") or {}
        name = md.get("name") or "Product"
        url  = md.get("product_url") or md.get("image_url") or ""
        lines.append(f"{i}) {name}" + (f"\n{url}" if url else ""))
    return "Here are visually similar items:\n\n" + "\n\n".join(lines)
