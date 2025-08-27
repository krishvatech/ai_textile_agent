import os
import httpx

META_TOKEN = os.getenv("META_CLOUD_API_TOKEN") or os.getenv("WHATSAPP_TOKEN") or ""
GRAPH_BASE = os.getenv("GRAPH_BASE", "https://graph.facebook.com")
GRAPH_VER  = os.getenv("GRAPH_VER", "v20.0")

if not META_TOKEN:
    raise RuntimeError("Set META_CLOUD_API_TOKEN (or WHATSAPP_TOKEN)")

def _auth_headers():
    return {"Authorization": f"Bearer {META_TOKEN}"}

async def get_media_url_and_meta(media_id: str) -> dict:
    url = f"{GRAPH_BASE}/{GRAPH_VER}/{media_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=_auth_headers())
        r.raise_for_status()
        return r.json()  # {"url": "...", "mime_type": "...", ...}

async def download_media_bytes(media_id: str) -> bytes:
    meta = await get_media_url_and_meta(media_id)
    media_url = meta.get("url")
    if not media_url:
        raise RuntimeError(f"No media URL for media_id={media_id}")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(media_url, headers=_auth_headers())
        r.raise_for_status()
        return r.content
