import asyncio
from openai import AsyncOpenAI
import json
import logging
import os
from typing import Tuple
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GPT_API_KEY")
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

**Textile Business Intents**:
1. **product_search** - Looking for clothes (saree, lehenga, kurti, suit, etc.)
2. **price_inquiry** - Asking about price, cost, budget
3. **color_preference** - Mentioning specific colors
4. **size_query** - Size, measurement, fitting questions
5. **fabric_inquiry** - Cotton, silk, georgette, chiffon, etc.
6. **order_placement** - Ready to buy, place order
7. **order_status** - Check existing order status
8. **catalog_request** - Want to see catalog, images, new designs, collections, what's available
9. **availability_check** - Stock availability
10. **customization** - Tailoring, alterations, custom design
11. **delivery_inquiry** - Delivery time, location, charges
12. **payment_query** - Payment methods, EMI, refund
13. **discount_inquiry** - Offers, deals, discounts
14. **rental_inquiry** - Rental services, rental price
15. **greeting** - Hello, hi, namaste
16. **complaint** - Problems, issues, returns
17. **other** - Anything else

**CRITICAL: Price vs Quantity Recognition Patterns**:
**GUJARATI PATTERNS**:
- **Price**: "500 ni", "₹500 na", "500 ના ભાવે", "500 rate", "500 કિંમત"
- **Quantity**: "1000 joia", "1000 જોઈએ", "1000 પીસ", "1000 sari", "જથ્થો"
**HINDI PATTERNS**:
- **Price**: "500 ka", "500 के", "₹500 में", "500 रुपए", "500 दाम"
- **Quantity**: "1000 chahiye", "1000 चाहिए", "1000 pieces", "1000 साड़ी"
**ENGLISH PATTERNS**:
- **Price**: "₹500 each", "500 rupees", "at 500", "price 500"
- **Quantity**: "1000 pieces", "need 1000", "want 1000", "1000 sarees"
**MIXED LANGUAGE PATTERNS**:
- "500 ni 1000 joia" = price: 500, quantity: 1000
- "1000 saree 500 ka" = quantity: 1000, price: 500
- "500 rate ma 1000 pieces" = price: 500, quantity: 1000

**Entity Extraction Guidelines**:
- **product**: Type of clothing mentioned
- **color**: Always normalize color names to their standard English equivalent regardless of language or script.
  For example:
  - "lal", "लाल", "લાલ" → "red"
  - "hara", "हरा", "હરું" → "green"
  - "pila", "पीला", "પીળું" → "yellow"
  - "neela", "नीला", "વાદળી" → "blue"
  If unsure, pick the closest standard English color.

- **fabric**: Recognize fabrics including regional or colloquial names like "resmi", "surti", "kanjeevaram", "mulmul", "mashru", etc.
  Normalize these to common fabric types:
  - "resmi" → "silk"
  - "surti" → "silk"
  - "mulmul" → "cotton"
  - "mashru" → "silk-cotton blend"
  Return the closest standard fabric if unsure.

- **price**: Unit price/rate per piece (NOT total amount)
Example:
    Input: "Find me red gorgette dress with large size for rent for wedding 2 in 10000"
    (Assume is_rental = 'true')
    Output: ( "price": 0, "rental_price": 10000 )

- **rental_price**: Rent price/rate per peace
Example:
    Input: "Red anarkali suit, Rs. 2500"
    (Assume is_rental = 'false')
    Output: ( "price": 2500, "rental_price": 0 )



- **size**: Recognize any mention of size or measurement using full names, abbreviations, numbers, or common variants (such as "small," "S," "medium," "M," "large," "L," "extra large," "XL," "free size," "freesize," "universal," "kids," "children," etc.).
Normalize all values to the following standardized forms:
  - "small", "sm", "s" → "S"
  - "medium", "med", "m", "midium" → "M"
  - "large", "l", "big" → "L"
  - "extra large", "xl", "x large", "bada", "double large" → "XL", "એક્સએલ" → "XL", "एक्सेल" → "XL"
  - "extra extra large", "xxl", "xx large" → "XXL","ડબલ એક્સેલ" → "XXL", "डबल एक्सेल" → "XXL" 
  Saree Size Normalization Rule:
  - If the product's category or name is "saree" (regardless of language/script or spelling), always set the extracted size to "Freesize".
  - Override any other size provided in the message (even if a different size like "XL", "M", or "kids" is mentioned with saree).
  For all non-saree products, apply standard size normalization rules.
  - "kids", "children", "child" → "Child"
  - If unsure or a numeric measurement is given (e.g., "42", "36", "chest 40"), return the exact value as size.
  For sarees, only use "Freesize".
  Return the closest standard size if the intent is clear.


- **occasion**: Recognize occasion-related words, including regional, slang, or colloquial expressions (such as "shaadi," "shadi," "fere," "biye," "vivaah," "pary," "daily wear," "navratri," "pooja," etc.).
  - "shaadi," "shadi," "vivaah," "biye," "wedding ceremony," "fera," "fere," "sadi," "saadi," "dulhan" → "wedding"
  - "reception," "party," "sangeet," "birthday," "anniversary" → "party"
  - "festival," "navratri," "diwali," "holi," "eid," "pooja," "raksha bandhan" → "festival"
  - "regular," "daily wear," "office," "work," "casual," "rozaana" → "casual"
  - "haldi," "mehendi" → "wedding" (since they are wedding sub-events)
Return the closest standard occasion (wedding, party, festival, casual) if unsure.

- **quantity**: Number of pieces desired
- **location**: Delivery location if mentioned

- **is_rental**: Recognize rental-related words
  - English: "for rent," "on rent," "rental," "rented," "rent price," "renting," "rent available"
  - Hindi: "किराए पर," "किराया," "रेंटल," "उधार पर," "भाड़ा"
  - Gujarati: "કિરાયે પર," "દાડા પર," "ભાડે," "રેન્ટ માટે"
  
  Example buy/purchase words:
  - English: "buy", "purchase", "want to buy", "order", "purchase price"
  - Hindi: "खरीदना", "लेना है", "खरीदेंगे", "ऑर्डर", "खरीद", "खरीदना है"
  - Gujarati: "ખરીદવું છે", " ઓર્ડર", "લૈસ", " ખરીદી"
  
  EXAMPLES:
    "Can I get these sarees for rent?" → is_rental: 'true'
    "I want to buy 10 lehengas" → is_rental: 'false'
    "I want dress" → is_rental:None
    "Send me your catalog" → is_rental: None
    "કિראયે પર લોંગા?" → is_rental: 'true'
    "મને ઓર્ડર કરવું છે" → is_rental: 'false'
  
  Always return is_rental as one of:
      'true' (for rental inquiry)
      'false' (for buy/purchase inquiry)
      None (if cannot determine)
  
  
**Customer Message**: "{text}"
**Detected Language**: "{detected_language}"

**Entity Extraction Guidelines**:
- **category**: Product category (Saree, Lehenga, Kurti, Suit, etc.)
- **fabric**: Fabric type (Silk, Cotton, Georgette, Chiffon, etc.)
- **color**: Specific colors mentioned
- **size**: Size requirements (Free Size, XL, L, etc.)
- **price**: Unit price/rate per piece (NOT total amount)
- **rental_price**: Rental price if mentioned
- **quantity**: Number of pieces needed
- **location**: Delivery location
- **occasion**: Wedding, party, casual, festival
- **is_rental**: true/false if rental inquiry

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

**EXAMPLES**:

Message: "lal sari 500 ni 1000 joia"
Output: {{
  "intent": "product_search",
  "entities": {{
    "product": "sari",
    "color": "red",
    "price": "500",
    "quantity": "1000"
  }},
  "confidence": 0.90,
  "is_question": false
}}

Message: "मुझे 1000 साड़ी चाहिए 500 के रेट में"
Output: {{
  "intent": "product_search",
  "entities": {{
    "product": "साड़ी",
    "price": "500",
    "quantity": "1000"
  }},
  "confidence": 0.90,
  "is_question": false
}}

Message: "1000 pieces saree at 500 rate"
Output: {{
  "intent": "product_search",
  "entities": {{
    "product": "saree",
    "price": "500",
    "quantity": "1000"
  }},
  "confidence": 0.90,
  "is_question": false
}}

Message: "લાલ સાડી ₹500 ના ભાવે 2000 જોઈએ છે"
Output: {{
  "intent": "product_search",
  "entities": {{
    "product": "સાડી",
    "color": "red",
    "price": "₹500",
    "quantity": "2000"
  }},
  "confidence": 0.90,
  "is_question": false
}}

Message: "mane resmi saree 500 ni 1000 joia lal color"
Output: {{
  "intent": "product_search",
  "entities": {{
    "product": "saree",
    "fabric": "silk",
    "price": "500",
    "quantity": "1000",
    "color": "red"
  }},
  "confidence": 0.90,
  "is_question": false
}}

Message: "any new design?"
Output: {{
  "intent": "catalog_request",
  "entities": {{}},
  "confidence": 0.90,
  "is_question": true
}}

Message: "kai navu che"
Output: {{
  "intent": "catalog_request",
  "entities": {{}},
  "confidence": 0.90,
  "is_question": true
}}
"""

    try:
        resp = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            # Do not send temperature/max_tokens with gpt-5-mini
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