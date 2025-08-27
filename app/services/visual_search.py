from __future__ import annotations
import os, io
from typing import Any, Dict, List, Optional
from PIL import Image
import torch, open_clip
from pinecone import Pinecone
import requests
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX", "textile-products")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ns1")
MODEL_NAME_IMAGE   = "ViT-B-32-quickgelu"
WHATSAPP_PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")  
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")

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

def _build_filter(tenant_id: Optional[int]) -> Optional[Dict[str, Any]]:
    f: Dict[str, Any] = {}
    if tenant_id is not None:
        f["tenant_id"] = {"$eq": int(tenant_id)}
    return f or None

def visual_search_bytes_sync(image_bytes: bytes, *, tenant_id: Optional[int] = None,
                             top_k: int = 12) -> List[Dict[str, Any]]:
    index, _, _ = _ensure_backends()
    vec = embed_image_bytes(image_bytes)
    res = index.query(
        vector=vec, top_k=int(top_k), include_metadata=True,
        namespace=PINECONE_NAMESPACE, filter=_build_filter(tenant_id)
    )
    return (res or {}).get("matches") or []

def format_matches_for_whatsapp(matches: List[Dict[str, Any]], limit: int = 5) -> str:
    if not matches:
        return "No visually similar items found."
    lines = []
    for i, m in enumerate(matches[:limit], 1):
        md = m.get("metadata") or {}
        name = md.get("name") or "Product"
        url  = md.get("image_url") or ""
        lines.append(f"{i}) {name}" + (f"\n{url}" if url else ""))
    return "Here are visually similar items:\n\n" + "\n\n".join(lines)

def format_matches_for_whatsapp_images(matches: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Build WhatsApp 'image' messages from Pinecone matches.
       Each item: {"type":"image","image":{"link": <image_url>, "caption": "<name>\\n<product_url>"}}.
    """
    out: List[Dict[str, Any]] = []
    for m in matches[:limit]:
        md = (m or {}).get("metadata") or {}
        img_url = md.get("image_url")
        if not img_url:
            continue
        caption = f"{md.get('name') or 'Product'}"
        if md.get("product_url"):
            caption += f"\n{md['product_url']}"
        out.append({
            "type": "image",
            "image": {"link": img_url, "caption": caption}
        })
    if not out:
        return [{"type": "text", "text": {"body": "No visually similar items found."}}]
    return out


def send_whatsapp_messages(to_wa_id: str, messages: List[Dict[str, Any]]) -> None:
    """Send a list of WhatsApp messages (image/text) to one recipient."""
    if not (WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_TOKEN):
        raise RuntimeError("Set WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_TOKEN")
    url = f"https://graph.facebook.com/v20.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    base = {"messaging_product": "whatsapp", "to": to_wa_id}
    for msg in messages:
        payload = {**base, **msg}
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()