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
    Process all entities and return complete entity structure with both values and None
    """
    # Define all possible entities based on your product schema
    all_entities = {
        "category": None,
        "fabric": None,
        "color": None,
        "size": None,
        "price": None,
        "rental_price": None,
        "quantity": None,
        "location": None,
        "occasion": None,
        "is_rental": None
        # "type": None    # <-- NEW FIELD
    }
    
    # Update with extracted entities, cleaning empty values
    for key, value in entities.items():
        if key in all_entities:
            if value and value not in [None, "", "null", "None", "N/A", "n/a"]:
                if isinstance(value, str) and value.strip():
                    all_entities[key] = value.strip()
                elif not isinstance(value, str):
                    all_entities[key] = value
           
    return all_entities
    
    # return all_entities

async def detect_textile_intent_openai(text: str, detected_language: str,allowed_categories: Optional[List[str]] = None) -> Tuple[str, dict, float]:
    """
    Detect customer intent for textile business using OpenAI
    """
    prompt = f"""
You are an AI assistant for a textile business in India specializing in wholesale and retail.

Goal: Analyze the customer message and return a single intent plus normalized entities.

Intents (lowercase values):
- greeting — short salutations/pleasantries only.
- product_search — message explicitly names a product category (garment/product type), optionally with attributes (fabric, color, size, price, quantity, location, occasion, rental).
- asking_inquiry — message asks about availability/options/prices/price range/starting price/rental price, with or without a category, and is NOT a request to show items.
- other — order status, tracking, delivery, payment, returns, complaints, small talk, or unrelated topics.

Decision rules (apply in order; pick exactly one):
1) If a product category is explicitly present → product_search.
2) Else if the message is primarily about availability/options/prices/rental without a clear category → asking_inquiry.
3) Else if it is only a salutation → greeting.
4) Else → other.
Never return "other" when rule 1 or 2 matches.

Entity extraction guidelines (normalize; be conservative; use null when unknown):
- product: Type of clothing mentioned; if identified, mirror it to "category" as well.
- category: The product/garment type  **Must be normalized to one of {allowed_categories}**
- color: Normalize to standard English color names regardless of language/script; if uncertain, choose the closest standard color.
- fabric: Normalize regional/colloquial names to common fabric types (e.g., standard silk/cotton/blends). If uncertain, choose the closest standard fabric.
- price: Unit price/rate per piece (number). If a numeric price is given and the request is about buying, set here; else leave null.
- rental_price: Rent price/rate per piece (number). If a numeric price is given and rental intent is explicit, set here; else leave null.
- size: Recognize full names, abbreviations, numbers, or measurements; normalize to one of "S","M","L","XL","XXL","Child", or preserve numeric (e.g., "42"). 
  Saree rule: if the category/name is "saree" (any spelling/script), always set size = "Freesize" and ignore other size mentions.
- occasion: Normalize to one of "wedding","party","festival","casual" when detectable; else null.
- quantity: Number of pieces desired (number) if stated; else null.
- location: Delivery location if stated; else null.
- is_rental: Only set true if rental intent is explicit; set false only if buy/purchase intent is explicit; otherwise null. Do not infer false for neutral queries.

Question detection:
- is_question = true if the user asks/requests info (including imperative forms like “find/show/get”); otherwise false.

Output requirements:
- Strict JSON only (no extra text).
- Use lowercase for "intent" values.
- "confidence" is a float between 0.0–1.0.
- Use numbers for price/rental_price/quantity; use booleans for is_rental; use null for unknown fields.
- Provide both "intent" and "intent_type" (set to the same value).

Customer Message: "{text}"
Detected Language: "{detected_language}"

Return exactly this JSON shape:
{{
    "intent": "",
    "intent_type": "",
    "entities": {{
        "product": null,
        "category": null,
        "fabric": null,
        "color": null,
        "size": null,
        "price": null,
        "rental_price": null,
        "quantity": null,
        "location": null,
        "occasion": null,
        "is_rental": null
    }},
    "confidence": 0.0,
    "is_question": false
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