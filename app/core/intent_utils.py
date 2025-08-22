import asyncio
from openai import AsyncOpenAI
import json
from typing import Tuple, List, Optional
import logging
import os
from typing import Tuple
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GPT_API_KEY")
gpt_model = os.getenv("GPT_MODEL")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

client = AsyncOpenAI(api_key=api_key)

def process_all_entities(entities: dict) -> dict:
    """
    Process all entities and return complete entity structure with both values and None.
    Preserve booleans (False) and numeric zeros.
    """
    all_entities = {
        "name": None,
        "category": None,
        "fabric": None,
        "color": None,
        "size": None,
        "price": None,
        "rental_price": None,
        "quantity": None,
        "location": None,
        "occasion": None,
        "is_rental": None,
        "start_date": None,
        "end_date": None,
        # "type": None  # keep if you’re using it
    }

    for key, value in (entities or {}).items():
        if key not in all_entities:
            continue

        # Keep booleans/numbers as-is (False/0 are valid)
        if isinstance(value, (bool, int, float)):
            all_entities[key] = value
            continue

        # Normalize strings; drop blanks and null-y text values
        if isinstance(value, str):
            v = value.strip()
            if v and v.lower() not in {"null", "none", "n/a"}:
                all_entities[key] = v
            continue

        # Fallback: keep other non-None types (e.g., lists/dicts) if truthy or explicitly non-empty
        if value is not None:
            all_entities[key] = value

    return all_entities


async def detect_textile_intent_openai(text: str, detected_language: str,allowed_categories: Optional[List[str]] = None,allowed_fabric: Optional[List[str]] = None,allowed_color: Optional[List[str]] = None,allowed_occasion: Optional[List[str]] = None) -> Tuple[str, dict, float]:
    """
    Detect customer intent for textile business using OpenAI
    """
    prompt = f"""
You are an AI assistant for a textile business in India specializing in wholesale and retail.
Goal: Analyze the customer message and return a single intent plus normalized entities.

Intents (lowercase values):
- greeting — short salutations/pleasantries only.
- product_search — message explicitly names a product category (garment/product type), optionally with attributes (fabric, color, size, price, quantity, location, occasion, rental).
- availability_check — user asks about date availability / booking / reserve for a specific product that is already selected in Context (e.g., product_variant_id present or exactly one selected item).
- asking_inquiry — ONLY when the user asks about availability/options/prices/etc. AND none of rules (-1, 0, 1, 2, 3, 3a, or 4) match. 
If any product attribute (including occasion) is present, do NOT return asking_inquiry — return product_search instead.
- website_inquiry — message contains a URL to one of our store domains and refers to a specific website/product/page. Treat ANY Shopify product/collection URL (https://*.myshopify.com/products/... or /collections/...) as our store.
- confirmation — user explicitly confirms proceeding to BUY or RENT the currently selected product (e.g., “confirm this”, “book it”, “I’ll take it”, “order now”, “reserve kar do”, “haan confirm”). This is a final go-ahead, not just a preference update.
- other — order status, tracking, delivery, payment, returns, complaints, small talk, or unrelated topics.

Decision rules (apply in order; pick exactly one):
- (-1) CONFIRMATION LOCK (highest priority):
   If the message is an explicit confirmation to proceed with BUY or RENT of the product that is already selected/in focus (e.g., context has product_variant_id, or there is exactly one selected item, or the user is replying “yes/confirm/book it” to the bot’s confirmation prompt), then:
   → intent = confirmation
   → set "confirmation": "yes" at the top level
   → set entities.is_rental = true if rental terms are explicitly mentioned (e.g., “on rent / kiraye par / rent pe / book for rent / haa”); set entities.is_rental = false if buy/purchase is explicit; otherwise leave null.
   → carry forward category (and other known filters) from Context when obvious; only fill what the user said now if new.
   → if dates/quantity are mentioned, populate start_date/end_date/quantity accordingly.
   This rule OVERRIDES all other rules, including the product-search lock below.


0) PRODUCT-SEARCH LOCK (highest after -1):
   If the message contains ANY product attribute — name, category, fabric, color, size, or occasion —
   OR explicit buy intent (e.g., “buy, purchase, kharidna, kharidvu, muje chahiye”)
   OR explicit rental intent (e.g., “on rent / rent pe / kiraye par / bhade” or is_rental=true/false),
   then → intent = "product_search".
   This OVERRIDES rules 1, 2, 4, 6 and 7 (unless -1 fired).
   Additionally: when buy is expressed, set is_rental=false; when rent is expressed, set is_rental=true.


1) If a product category is explicitly present → product_search.

2) If the message contains an http(s) URL that points to our store (including any Shopify product/collection URL like https://*.myshopify.com/products/... or .../collections/...) → website_inquiry,
   UNLESS rule (-1) or 0 already fired.

3) If this message is ONLY a refinement (e.g., “on rent / rent pe / kiraye par / i want rent / i want on rent / muje kiraye pe chahiye”, buy/purchase, color, fabric, size, occasion, budget)
   AND the Context above already contains a non-null category, then → product_search (continue browsing with updated filters). Do NOT switch to asking_inquiry in this case.
   For rental refinements, set is_rental=true in entities.
   Example: If context has "category": "saree" and message is "muje kiraye pe chahiye" → product_search, with category carried over.

3a) If the message matches a refinement (as in rule 3) but there is NO non-null category in context,
    infer a default category from allowed_categories (use the first one if available, e.g., "saree") and set intent to product_search.
    Only do this if allowed_categories is non-empty; else fall to rule 6.

4) If the message includes a calendar date or booking phrasing (e.g., '24 Aug', 'aaj/today', 'kal/tomorrow', 'ke liye'), return availability_check and populate start_date/end_date,
   UNLESS rule (-1) or 0 already fired. If no specific product is selected, still return availability_check (the app will ask the user to pick a product).

5) Else if it is only a salutation → greeting.

6) Else if the message is primarily about availability/options/prices/rental without a clear category → asking_inquiry.

7) Else → other.

Never return "other" when rule (-1), 0, 1, 2, 3, 3a, or 4 matches.

Entity extraction guidelines (normalize; be conservative; use null when unknown):
- product: Natural product name mentioned (freeform). If identified, also mirror its garment type to "category".
- name: Natural product name mentioned (freeform), and mirror its garment type to "category".
- category: The product/garment type  **Must be normalized to one of {allowed_categories}**.
  For ambiguous terms like "choli", prefer "chaniya choli" if context fits or is a closer cultural match.
- color: The product/garment color **Must be normalized to one of {allowed_color}**. Normalize to standard English color names regardless of language/script; if uncertain, choose the closest standard color.
- fabric: The product/garment fabric **Must be normalized to one of {allowed_fabric}**. Normalize regional/colloquial names to common fabric types (e.g., standard silk/cotton/blends). If uncertain, choose the closest standard fabric.
- price: Unit price/rate per piece (number). If a numeric price is given and the request is about buying, set here; else null.
- rental_price: Rent price/rate per piece (number). If a numeric price is given and rental intent is explicit, set here; else null.
- size: Recognize full names, abbreviations, numbers, or measurements; normalize to one of "S","M","L","XL","XXL","Child", or preserve numeric (e.g., "42").
  Saree rule: if the category/name is "saree" (any spelling/script), always set size = "Freesize" and ignore other size mentions.
- occasion: The product occasion **Must be normalized to one of {allowed_occasion}**. Normalize to standard English occasion names regardless of language/script; if uncertain, choose the closest standard occasion
- quantity: Number of pieces desired (number). Convert digits or number-words into an integer. Handle English / Hinglish / Hindi / Gujarati in native or roman scripts. Output quantity as a JSON number (not a string). If absent or ambiguous, set null.
- location: Delivery location if stated; else null.
- is_rental: Only set true if rental intent is explicit in that detected language; set false only if buy/purchase intent is explicit; otherwise null. Do not infer false for neutral queries.
- start_date: If message mentions a booking/need date (absolute or relative like "aaj/today", "kal/tomorrow"), normalize to YYYY-MM-DD; else null.
- end_date: If a range/return date is mentioned, normalize to YYYY-MM-DD; else null. If only a single date is given, set end_date = start_date.
- type: Sub-type or variant of the category (e.g., "lehenga" for a choli variant, "banarasi" for saree). Normalize to standard terms from allowed_categories if possible; else null. Be conservative—only set if explicitly implied.

Question detection:
- is_question = true if the user asks/requests info (including imperative forms like “find/show/get” and availability/booking queries); otherwise false.

Output requirements:
- Strict JSON only (no extra text).
- Use lowercase for "intent" values.
- "confidence" is a float between 0.0–1.0.
- Use numbers for price/rental_price/quantity; use booleans for is_rental; use null for unknown fields.
- Add a TOP-LEVEL field "confirmation": set to "yes" ONLY when the message is an explicit confirmation to proceed with buy/rent as per the Confirmation Lock; otherwise null.
- Provide both "intent" and "intent_type" (set to the same value).

Customer Message: "{text}"
Detected Language: "{detected_language}"

Return exactly this JSON shape:
{{
    "intent": "",
    "intent_type": "",
    "entities": {{
        "name": null,
        "category": null,
        "fabric": null,
        "color": null,
        "size": null,
        "price": null,
        "rental_price": null,
        "quantity": null,
        "location": null,
        "occasion": null,
        "is_rental": null,
        "start_date": null,
        "end_date": null,
        "type": null  // <-- keep this
    }},
    "confidence": 0.0,
    "is_question": false,
    "confirmation": null   // <-- set to "yes" when intent = "confirmation"
}}
"""
    try:
        resp = await client.chat.completions.create(
            model=gpt_model,
            messages=[{"role": "user", "content": prompt}],
            # Do not send temperature/max_tokens with gpt_model
        )
        content = resp.choices[0].message.content.strip()
        # Clean possible code fences
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        processed_entities = process_all_entities(result.get("entities", {}))
        filtered_entities = clean_price_fields(processed_entities)
        return (
            result.get("intent", "other"),
            filtered_entities,
            float(result.get("confidence", 0.1))
        )
    except Exception as e:
        logging.error(f"Intent detection failed: {e}")
        empty_entities = process_all_entities({})
        return "other", empty_entities, 0.1



def clean_price_fields(entities: dict) -> dict:
    """
    Removes 'price' or 'rental_price' keys entirely from entities based on is_rental.
    - If is_rental is True, remove 'price'.
    - If is_rental is False, remove 'rental_price'.
    - If is_rental is None or unclear, keep both.
    """
    is_rental_val = entities.get('is_rental')
    is_rental_is_true = (is_rental_val is True or (isinstance(is_rental_val, str) and str(is_rental_val).lower() == "true"))
    is_rental_is_false = (is_rental_val is False or (isinstance(is_rental_val, str) and str(is_rental_val).lower() == "false"))

    # Defensive copy so as not to mutate original
    filtered = dict(entities)
    if is_rental_is_true:
        filtered.pop('price', None)
    elif is_rental_is_false:
        filtered.pop('rental_price', None)
    # Otherwise, keep both if intent is unclear
    return filtered



def format_entities(entities: dict) -> str:
    """
    Format entities for display, showing both populated and None values
    """
    if not entities:
        return "None"
    
    formatted_lines = []
    for k, v in entities.items():
        if v is not None:
            formatted_lines.append(f" ✓ {k}: {v}")
        else:
            formatted_lines.append(f" ○ {k}: None")
    
    return "\n".join(formatted_lines)
