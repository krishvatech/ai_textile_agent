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
- stock_check — user replies with ONLY a quantity (e.g., "1", "2", "25", "2 pcs"), meaning “check if this many of the currently selected item/category is available right now.”
- other — order status, tracking, delivery, payment, returns, complaints, small talk, or unrelated topics.

Decision rules (apply in order; pick exactly one):

0) CONFIRMATION LOCK (highest priority):
   If the message is an explicit confirmation to proceed with BUY or RENT of the product that is already selected/in focus (e.g., context has product_variant_id, or there is exactly one selected item, or the user is replying “yes/confirm/book it” to the bot’s confirmation prompt), then:
   → intent = confirmation
   → set "confirmation": "yes" at the top level
   → set entities.is_rental = true if rental terms are explicitly mentioned (e.g., “on rent / kiraye par / rent pe / book for rent / haa”); set entities.is_rental = false if buy/purchase is explicit; otherwise leave null.
   → carry forward category (and other known filters) from Context when obvious; only fill what the user said now if new.
   → if dates/quantity are mentioned, populate start_date/end_date/quantity accordingly.
   This rule OVERRIDES all other rules.

1) PRODUCT-SEARCH LOCK (highest after 0):
   If the message contains ANY product attribute — name, category, fabric, color, size, or occasion —
   OR explicit buy intent (e.g., “buy, purchase, kharidna, kharidvu, muje chahiye”)
   OR explicit rental intent (e.g., “on rent / rent pe / kiraye par / bhade” or is_rental=true/false),
   then → intent = "product_search".
   This OVERRIDES rules 2, 3, 5, 7 and 8 (unless 0 or 1a fired).
   Additionally: when buy is expressed, set is_rental=false; when rent is expressed, set is_rental=true.

1a) OPTIONS-ONLY QUESTION EXCEPTION (overrides 1 for list-style questions):
   If the message is a WH / option-list question about available attributes WITHOUT choosing one,
   e.g., “which fabrics do you have in kurta?”, “what colors are available in saree?”,
   “kaun-se sizes milenge kurta me?”, “कुर्ता में कौन-कौन से कलर हैं?”, “કુર્તામાં કયા કલર્સ છે?”,
   and it does NOT explicitly give a concrete value (like “cotton”, “red”, “XL”) and does NOT express buy/rent,
   then → intent = "asking_inquiry".
   Populate entities.category if a garment type is named; keep is_rental = null.

1b) PRICE-ONLY OVERRIDE:
   If the message is primarily a price question (e.g., “starting price of …”, “price range for …”, “what’s the price of …”, “kitna shuru hota hai …”, “rent ka rate kya hai …”)
   AND it mentions a category
   AND it does NOT add any other attribute (no color/fabric/size/occasion) and does NOT explicitly confirm buy/rent,
   then → intent = asking_inquiry.
   Populate entities.category from the message, set entities.is_rental only if explicitly stated, leave price/rental_price null.
   This rule OVERRIDES Rule 1.

1c) ATTRIBUTE-LIST OVERRIDE:
   If the user is asking for available options of a single attribute — e.g.
   “which fabrics do you have”, “what colors are available”, “what sizes do you carry”,
   “categories you have”, “kaun-kaun se fabric hai?”, “konsa kapda milta hai?” —
   then → intent = asking_inquiry.
   Behavior:
     • Keep all existing context fields unchanged (category/is_rental/occasion/etc.).
     • Set asked_now to exactly that attribute (e.g., ["fabric"]).
     • Do NOT ask for rental dates here, even if is_rental = true.
     • The answer should list unique values filtered by tenant + current context.
   This rule OVERRIDES Rule 1.

1d) QUANTITY-ONLY OVERRIDE (→ stock_check):
   If the message contains ONLY a quantity (digits or number-words) with optional light fillers/units
   like "pcs", "piece(s)", "qty", "please/pls", and NO other attribute (no color/fabric/size/occasion/price,
   no buy/rent words, no URL), AND there is a product in focus (context has product_variant_id OR a non-null category),
   then → intent = stock_check.
   Behavior:
     • Set entities.quantity to the parsed number (as a JSON number).
     • Carry forward category (and other known filters) from context; do not change them here.
     • Leave price/rental_price null.
   This rule OVERRIDES Rule 1.

1e) PROCEED-TO-STOCK OVERRIDE (→ stock_check):
   If the user expresses a clear decision to take/buy/rent the item currently in focus — e.g., “I want this”, “yeh chahiye”, “isko le lo”, “I want buy this saree”, “isse hi de do”, or equivalent in any language — then treat it as an immediate stock check.
   Conditions:
     • There is a product in focus (context has product_variant_id OR exactly one selected item OR a non-null category).
   Behavior:
     • intent = stock_check
     • entities.is_rental: set false if buy wording is explicit; true if rent wording is explicit; else leave as-is/null.
     • entities.quantity: if a number is present, set it; otherwise default to 1.
   This rule OVERRIDES Rules 1, 4 and 4a, and takes precedence over Rule 0 only when the message is NOT an explicit confirmation (“confirm/book/pay”).

2) If a product category is explicitly present → product_search, UNLESS Rule 0 or 1a/1b/1c/1d/1e already fired.

3) If the message contains an http(s) URL that points to our store (including any Shopify product/collection URL like https://*.myshopify.com/products/... or .../collections/...) → website_inquiry,
   UNLESS Rule 0 or 1 already fired.

4) If this message is ONLY a refinement (e.g., “on rent / rent pe / kiraye par / i want rent / i want on rent / muje kiraye pe chahiye”, buy/purchase, a specific color/fabric/size value, occasion, budget)
   AND the Context above already contains a non-null category, then → product_search (continue browsing with updated filters).
   (Exception: if the refinement is an option-list WH question that doesn’t select a concrete value, use asking_inquiry as per 1a/1c.)

4a) If the message matches a refinement (as in Rule 4) but there is NO non-null category in context,
    infer a default category from allowed_categories (use the first one) and set intent to product_search,
    UNLESS Rule 1a applies (then asking_inquiry with that inferred category).

5) If the message includes a calendar date or booking phrasing (e.g., '24 Aug', 'aaj/today', 'kal/tomorrow', 'ke liye'), return availability_check and populate start_date/end_date,
   UNLESS Rule 0 or 1 already fired. If no specific product is selected, still return availability_check (the app will ask the user to pick a product).

6) Else if it is only a salutation → greeting.

7) Else if the message is primarily about availability/options/prices/rental without a clear category → asking_inquiry.

8) Else → other.

- When asked_now ∈ {"fabric","color","size","category"}:
  • Provide a concise, comma-separated list of available options filtered by current context.
  • Ask the user to pick ONE.
  • Do NOT ask for start/end rental dates in this turn (dates come only after the user picks attributes or asks about availability).

Never return "other" when Rule 0, 1, 2, 3, 4, 4a, or 5 matches.


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
