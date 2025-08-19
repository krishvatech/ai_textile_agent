import os, json, logging, re
from typing import Dict, Any, List, Tuple
from openai import AsyncOpenAI

# OpenAI client + model
_CLIENT = AsyncOpenAI(api_key=os.getenv("GPT_API_KEY"))
_GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

ATTRIBUTE_DETECT_SYSTEM = """
You classify what the shopper is ASKING TO LIST right now for a textiles store.

Return STRICT JSON only (no prose). Do not include explanations.

Concepts:
- Listing asks (enumerations):
  * product_list        → browsing categories or “what do you have?”
  * fabric_list         → asking which fabrics are available
  * color_list          → asking which colors are available (optionally within a category)
  * size_list           → asking which sizes are available (optionally within a category)
  * occasion_list       → asking which OCCASIONS you carry (e.g., Wedding / Party / Festival / Casual/Office)
  * rental_list         → asking whether items are for Rent or Purchase (list both if applicable)
  * price_list          → asking price/range/starting price (purchase)
  * rental_price_list   → asking rental price/range/starting price (rent)

- Not listing: concrete filter requests (e.g., “red saree”, “cotton kurta sets”, “show lehenga”)
  These are requests to see items, not lists of options; they do NOT set any *_list flag.

Use the provided context (acc_entities) only as memory for implicit follow-ups.
Example: if acc_entities.category = "saree" and the user asks “what colors?”, treat it as color_list with slots.category="saree".

Rules:
1) asked_now must be a subset of: ["category","fabric","color","size","occasion","rental","price","rental_price"], mapping:
   - product_list      → "category"
   - fabric_list       → "fabric"
   - color_list        → "color"
   - size_list         → "size"
   - occasion_list     → "occasion"
   - rental_list       → "rental"
   - price_list        → "price"
   - rental_price_list → "rental_price"
2) If the utterance is generic browsing (“what do you have?”, “show items”), mark product_list=true and asked_now=["category"].
3) If multiple listing intents are clearly requested in one utterance, include all (rare).
4) If ambiguous between listing vs filter, prefer filter (i.e., no *_list flags).
5) Multilingual (en/hi/gu/hinglish). Use semantic understanding, not exact keywords.
6) Do not invent slots; fill only when explicit or trivially implied by context.

Output JSON:
{
  "asked_now": ["category" | "fabric" | "color" | "size" | "occasion" | "rental" | "price" | "rental_price"],
  "flags": {
    "product_list": bool,
    "fabric_list": bool,
    "color_list": bool,
    "size_list": bool,
    "occasion_list": bool,
    "rental_list": bool,
    "price_list": bool,
    "rental_price_list": bool
  },
  "slots": {
    "category": string|null,
    "fabric": string|null,
    "color": string|null,
    "size": string|null,
    "occasion": string|null,
    "rental": "Rent"|"Purchase"|null
  },
  "confidence": {
    "product_list": number, "fabric_list": number, "color_list": number, "size_list": number,
    "occasion_list": number, "rental_list": number, "price_list": number, "rental_price_list": number
  },
  "triggers": []  // optional short evidence copied from the user text; omit if none
}
Ensure asked_now is consistent with flags (e.g., if product_list=true then "category" must be in asked_now).
"""

def _build_messages(text: str, acc_entities: Dict[str, Any] | None) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ATTRIBUTE_DETECT_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"text": text or "", "context": {"acc_entities": acc_entities or {}}},
                ensure_ascii=False
            )
        },
    ]



# ---------- Public API ----------
async def detect_requested_attributes_async(
    text: str,
    acc_entities: Dict[str, Any] | None = None,
) -> List[str]:
    """
    Minimal list like ['category'] | ['fabric'] | ['color'] | ['size'] | ['occasion'] | ['rental'] | ['price'] | ['rental_price'].
    """
    try:
        resp = await _CLIENT.chat.completions.create(
            model=_GPT_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=_build_messages(text, acc_entities),
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        asked_now = data.get("asked_now") or []
        allowed = {"category", "fabric", "color", "size", "occasion", "rental", "price", "rental_price"}
        asked_now = [a for a in asked_now if a in allowed]
        return asked_now
    except Exception:
        logging.exception("detect_requested_attributes_async failed")
        return []

async def detect_requested_full_async(
    text: str,
    acc_entities: Dict[str, Any] | None = None,
) -> Tuple[List[str], Dict[str, bool], Dict[str, Any]]:
    """
    Returns (asked_now, flags, slots) with occasion/rental/price/rental_price included.
    """
    try:
        resp = await _CLIENT.chat.completions.create(
            model=_GPT_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=_build_messages(text, acc_entities),
        )
        data = json.loads(resp.choices[0].message.content or "{}")

        allowed = {"category","fabric","color","size","occasion","rental","price","rental_price"}
        asked_now = [a for a in (data.get("asked_now") or []) if a in allowed]

        flags = {
            "product_list":       bool((data.get("flags") or {}).get("product_list")),
            "fabric_list":        bool((data.get("flags") or {}).get("fabric_list")),
            "color_list":         bool((data.get("flags") or {}).get("color_list")),
            "size_list":          bool((data.get("flags") or {}).get("size_list")),
            "occasion_list":      bool((data.get("flags") or {}).get("occasion_list")),
            "rental_list":        bool((data.get("flags") or {}).get("rental_list")),
            "price_list":         bool((data.get("flags") or {}).get("price_list")),
            "rental_price_list":  bool((data.get("flags") or {}).get("rental_price_list")),
        }
        slots = data.get("slots") or {}
        return asked_now, flags, slots
    except Exception:
        logging.exception("detect_requested_full_async failed")
        return (
            [],
            {
                "product_list": False, "fabric_list": False, "color_list": False,
                "size_list": False, "occasion_list": False, "rental_list": False,
                "price_list": False, "rental_price_list": False
            },
            {}
        )
