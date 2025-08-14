import asyncio
from openai import AsyncOpenAI
import json
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

async def detect_textile_intent_openai(text: str, detected_language: str) -> Tuple[str, dict, float]:
    """
    Detect customer intent for textile business using OpenAI
    """
    prompt = f"""
You are an AI assistant for a textile business in India specializing in wholesale and retail.
**Your Task**: Analyze the customer message and identify their business intent.
**greeting** – Short salutations or pleasantries.
**asking_inquiry**:
  If the user's message is primarily asking about available products, fabrics, colors, sizes, prices, rental options, or combinations of these — whether they use direct questions or indirect phrasing — classify the intent as **"asking_inquiry"**.
Do NOT classify as "asking_inquiry" if:
- The message is about order status, tracking, delivery, payment, or returns.
- The message is a greeting, thank you, or unrelated conversation.
**Customer Message**: "{text}"
**Detected Language**: "{detected_language}"
**Response Format** (JSON only):
{{
    "intent": "",
    "entities": {{
        "category": "",
        "fabric": "",
        "color": "",
        "size": "",
        "price": "",
        "rental_price": "",
        "quantity": "",
        "location": "",
        "occasion": "",
        "is_rental": ""
    }},
    "confidence": <0.0-1.0>,
    "is_question": <true/false>
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