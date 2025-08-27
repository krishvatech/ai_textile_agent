# #!/usr/bin/env python3
# # app/tests/visual_search.py
# # Query Pinecone (ai-textile-agent) using an image -> returns nearest text+image vectors.

# import os
# import re
# import json
# import argparse
# from io import BytesIO
# from hashlib import md5
# from typing import Any, Dict, List, Optional

# from dotenv import load_dotenv
# import requests
# from PIL import Image

# import torch
# import open_clip
# from pinecone import Pinecone

# # ============== ENV ==============
# load_dotenv()
# PINECONE_API_KEY ="pcsk_27uP34_94NTBKhAoJkJ7TjtJSw2isBjURag5vfHBTUoMANfrV6pQ5TXV1Dv68NkEhSrhxn"
# PINECONE_INDEX = "ai-textile-agent"
# PINECONE_NAMESPACE ="default"

# if not PINECONE_API_KEY:
#     raise RuntimeError("PINECONE_API_KEY not set")

# # ============== CLIP IMAGE MODEL (ViT-B-32-quickgelu) ==============
# MODEL_NAME_IMAGE = "ViT-B-32-quickgelu"
# _device = "cpu"
# try:
#     torch.set_num_threads(min(4, (os.cpu_count() or 4)))
# except Exception:
#     pass

# clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#     MODEL_NAME_IMAGE, pretrained="openai"
# )
# clip_model.eval().to(_device)

# # ============== PINECONE ==============
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(PINECONE_INDEX)

# # ============== HELPERS ==============
# def is_url(s: str) -> bool:
#     return bool(re.match(r"^https?://", s.strip(), re.I))

# def load_image_any(path_or_url: str) -> Image.Image:
#     if is_url(path_or_url):
#         resp = requests.get(path_or_url, timeout=30)
#         resp.raise_for_status()
#         return Image.open(BytesIO(resp.content)).convert("RGB")
#     else:
#         return Image.open(path_or_url).convert("RGB")

# def image_embed(img: Image.Image) -> List[float]:
#     with torch.inference_mode():
#         x = clip_preprocess(img).unsqueeze(0).to(_device)
#         feats = clip_model.encode_image(x)
#         feats = feats / feats.norm(dim=-1, keepdim=True)
#     return feats[0].cpu().numpy().astype("float32").tolist()  # 512-d

# def build_filter(tenant_id: Optional[int], modality: str) -> Optional[Dict[str, Any]]:
#     f: Dict[str, Any] = {}
#     if tenant_id is not None:
#         f["tenant_id"] = {"$eq": int(tenant_id)}
#     m = modality.strip().lower()
#     if m in {"image", "text"}:
#         f["modality"] = {"$eq": m}
#     # if "both", no modality filter (return mixed text+image hits)
#     return f or None

# def truncate(s: Any, n: int = 90) -> str:
#     s = "" if s is None else str(s)
#     return s if len(s) <= n else s[: n - 1] + "…"

# def pretty_print_matches(matches: List[Dict[str, Any]]):
#     if not matches:
#         print("No results.")
#         return
#     print("\nTop matches:")
#     for i, m in enumerate(matches, 1):
#         md = m.get("metadata") or {}
#         print(f"{i:>2}. score={m.get('score'):.4f}  id={m.get('id')}")
#         print(f"    modality: {md.get('modality')} | tenant: {md.get('tenant_id')} | "
#               f"product_id: {md.get('product_id')} | variant_id: {md.get('variant_id')}")
#         print(f"    name: {truncate(md.get('name'))}")
#         print(f"    category/type: {md.get('category')}/{md.get('type')} | "
#               f"fabric: {md.get('fabric')} | color: {md.get('color')} | size: {md.get('size')}")
#         if md.get("image_url"):
#             print(f"    image_url: {md.get('image_url')}")
#         if md.get("product_url"):
#             print(f"    product_url: {md.get('product_url')}")
#         print("")

# def query_by_image(path_or_url: str, tenant_id: Optional[int], modality: str, top_k: int):
#     img = load_image_any(path_or_url)
#     vec = image_embed(img)
#     pc_filter = build_filter(tenant_id, modality)

#     res = index.query(
#         vector=vec,
#         top_k=int(top_k),
#         include_metadata=True,
#         namespace=PINECONE_NAMESPACE,
#         filter=pc_filter
#     )

#     matches = (res or {}).get("matches") or []
#     pretty_print_matches(matches)

# # ============== CLI ==============
# def main():
#     parser = argparse.ArgumentParser(
#         description="Visual search in ai-textile-agent using an image (URL or local file)."
#     )
#     parser.add_argument("--image", "-i", required=True, help="Image URL or local path")
#     parser.add_argument("--tenant", "-t", type=int, default=None, help="Optional tenant_id filter")
#     parser.add_argument("--modality", "-m", choices=["both", "image", "text"], default="both",
#                         help="Restrict results to image-only, text-only, or both (default)")
#     parser.add_argument("--topk", "-k", type=int, default=12, help="Number of results to return")
#     args = parser.parse_args()

#     print(f"Index: {PINECONE_INDEX}  |  Namespace: {PINECONE_NAMESPACE}")
#     print(f"Query image: {args.image}")
#     if args.tenant is not None:
#         print(f"Filter tenant_id: {args.tenant}")
#     print(f"Modality: {args.modality}  |  top_k: {args.topk}")

#     query_by_image(args.image, args.tenant, args.modality, args.topk)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# local_visual_search.py
# Standalone local tester:
# - Loads an image (URL or local path)
# - Embeds with CLIP ViT-B-32-quickgelu (512-dim)
# - Queries your Pinecone index (text+image stored together)
# - Prints pretty results or JSON

import os
import re
import io
import json
import argparse
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from PIL import Image

import torch
import open_clip
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

# ============== ENV / DEFAULTS ==============
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DEFAULT_INDEX = os.getenv("PINECONE_INDEX", "ai-textile-agent")
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
MODEL_NAME_IMAGE = "ViT-B-32-quickgelu"

if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY in your environment.")

# ============== INIT: Pinecone + CLIP ==============
pc = Pinecone(api_key=PINECONE_API_KEY)

# (we’ll open the index after parsing args)
_device = "cpu"
try:
    torch.set_num_threads(min(4, (os.cpu_count() or 4)))
except Exception:
    pass

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME_IMAGE, pretrained="openai"
)
clip_model.eval().to(_device)

# ============== HELPERS ==============
_URL_RX = re.compile(r"^https?://", re.I)

def is_url(s: str) -> bool:
    return bool(_URL_RX.match((s or "").strip()))

def _default_referer_for(url: str) -> Optional[str]:
    try:
        p = urlparse(url)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}"
    except Exception:
        pass
    return None

def load_image_any(path_or_url: str, referer: Optional[str], ua: str) -> Image.Image:
    if is_url(path_or_url):
        headers = {"User-Agent": ua}
        if not referer:
            referer = _default_referer_for(path_or_url)
        if referer:
            headers["Referer"] = referer

        resp = requests.get(path_or_url, timeout=30, headers=headers)
        # Be explicit if remote host returns XML/HTML error instead of image
        ct = (resp.headers.get("Content-Type") or "").lower()
        if "image" not in ct:
            snippet = resp.text[:200].replace("\n", " ")
            raise RuntimeError(
                f"Remote returned non-image content. Status={resp.status_code} "
                f"Content-Type='{ct}' Body starts: {snippet!r}"
            )
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    # local path
    return Image.open(path_or_url).convert("RGB")

def image_embed(img: Image.Image) -> List[float]:
    with torch.inference_mode():
        x = clip_preprocess(img).unsqueeze(0).to(_device)
        feats = clip_model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype("float32").tolist()

def build_filter(tenant_id: Optional[int], modality: str,
                 category: Optional[str], color: Optional[str], fabric: Optional[str]) -> Optional[Dict[str, Any]]:
    f: Dict[str, Any] = {}
    if tenant_id is not None:
        f["tenant_id"] = {"$eq": int(tenant_id)}
    if category:
        f["category"] = {"$eq": category}
    if color:
        f["color"] = {"$eq": color}
    if fabric:
        f["fabric"] = {"$eq": fabric}
    return f or None

def group_matches_by_variant(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for m in matches:
        md = m.get("metadata") or {}
        key = (md.get("tenant_id"), md.get("product_id"), md.get("variant_id"))
        prev = best.get(key)
        if prev is None:
            best[key] = m
            continue
        # keep higher score; if equal, prefer image modality
        if m.get("score", 0) > prev.get("score", 0):
            best[key] = m
        elif m.get("score", 0) == prev.get("score", 0):
            if (md.get("modality") == "image") and ((prev.get("metadata") or {}).get("modality") != "image"):
                best[key] = m
    return list(best.values())

def to_compact_list(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in matches:
        md = m.get("metadata") or {}
        out.append({
            "score": m.get("score"),
            "id": m.get("id"),
            "modality": md.get("modality"),
            "tenant_id": md.get("tenant_id"),
            "product_id": md.get("product_id"),
            "variant_id": md.get("variant_id"),
            "name": md.get("name"),
            "category": md.get("category"),
            "type": md.get("type"),
            "fabric": md.get("fabric"),
            "color": md.get("color"),
            "size": md.get("size"),
            "image_url": md.get("image_url"),
            "product_url": md.get("product_url"),
        })
    return out

def truncate(s: Any, n: int = 90) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

def pretty_print_matches(matches: List[Dict[str, Any]]):
    if not matches:
        print("No results.")
        return
    print("\nTop matches:")
    for i, m in enumerate(matches, 1):
        md = m.get("metadata") or {}
        print(f"{i:>2}. score={m.get('score'):.4f}  id={m.get('id')}")
        print(f"    modality: {md.get('modality')} | tenant: {md.get('tenant_id')} | "
              f"product_id: {md.get('product_id')} | variant_id: {md.get('variant_id')}")
        print(f"    name: {truncate(md.get('name'))}")
        print(f"    category/type: {md.get('category')}/{md.get('type')} | "
              f"fabric: {md.get('fabric')} | color: {md.get('color')} | size: {md.get('size')}")
        if md.get("image_url"):
            print(f"    image_url: {md.get('image_url')}")
        if md.get("product_url"):
            print(f"    product_url: {md.get('product_url')}")
        print("")

# ============== MAIN QUERY ==============
def main():
    parser = argparse.ArgumentParser(description="Local visual search tester for ai-textile-agent.")
    parser.add_argument("--image", "-i", required=True, help="Image URL or local file path")
    parser.add_argument("--tenant", "-t", type=int, default=None, help="Optional tenant_id filter")
    parser.add_argument("--modality", "-m", choices=["both", "image", "text"], default="both",
                    help="Restrict results to image-only, text-only, or both (default: both)")
    parser.add_argument("--topk", "-k", type=int, default=12, help="Number of results to return")
    parser.add_argument("--category", default=None, help="Filter, e.g. Saree or Sherwani")
    parser.add_argument("--color", default=None, help="Filter, e.g. Red")
    parser.add_argument("--fabric", default=None, help="Filter, e.g. Silk")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-group", action="store_true",
                        help="Do not group by (tenant,product,variant); default groups to best per variant")
    parser.add_argument("--index", default=DEFAULT_INDEX, help=f"Pinecone index (default: {DEFAULT_INDEX})")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help=f"Pinecone namespace (default: {DEFAULT_NAMESPACE})")
    parser.add_argument("--referer", default=None, help="Optional Referer header for URL fetch")
    parser.add_argument("--ua", default="Mozilla/5.0", help="User-Agent for URL fetch")
    args = parser.parse_args()

    # Open Pinecone index
    index = pc.Index(args.index)

    print(f"Index: {args.index}  |  Namespace: {args.namespace}")
    print(f"Query image: {args.image}")
    if args.tenant is not None:
        print(f"Filter tenant_id: {args.tenant}")
    print(f"Modality: {args.modality}  |  top_k: {args.topk}")

    # Load + embed
    img = load_image_any(args.image, referer=args.referer, ua=args.ua)
    vec = image_embed(img)

    # Build filter & query
    pc_filter = build_filter(args.tenant, args.modality, args.category, args.color, args.fabric)
    res = index.query(
        vector=vec,
        top_k=int(args.topk),
        include_metadata=True,
        namespace=args.namespace,
        filter=pc_filter
    )
    matches = (res or {}).get("matches") or []

    # Grouping / output
    if not args.no_group:
        matches = group_matches_by_variant(matches)

    if args.json:
        print(json.dumps(to_compact_list(matches), ensure_ascii=False, indent=2))
    else:
        pretty_print_matches(matches)

if __name__ == "__main__":
    main()
