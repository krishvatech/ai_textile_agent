from openai import AsyncOpenAI
import json
import logging
from typing import Tuple
import os
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("GPT_API_KEY"))

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
14. **greeting** - Hello, hi, namaste
15. **complaint** - Problems, issues, returns
16. **other** - Anything else
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
- **product**: Type of clothing
- **color**: Specific colors mentioned
- **price_range**: Unit price/rate per piece (NOT total amount)
- **size**: Size requirements
- **fabric**: Fabric type
- **occasion**: Wedding, party, casual, festival
- **quantity**: Number of pieces needed
- **location**: Delivery location
**Customer Message**: "{text}"
**Detected Language**: "{detected_language}"
**Response Format** (JSON only):
{{
    "intent": "<intent_name>",
    "entities": {{
        "product": "<value_or_null>",
        "color": "<value_or_null>",
        "price_range": "<value_or_null>",
        "size": "<value_or_null>",
        "fabric": "<value_or_null>",
        "occasion": "<value_or_null>",
        "quantity": "<value_or_null>",
        "location": "<value_or_null>"
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
        "color": "lal",
        "price_range": "500",
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
        "price_range": "500",
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
        "price_range": "500",
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
        "color": "લાલ",
        "price_range": "₹500",
        "quantity": "2000"
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
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300  # Increased for better processing
        )
        content = response.choices[0].message.content.strip()
        # Clean JSON response
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        # :white_check_mark: FILTER OUT EMPTY ENTITIES
        filtered_entities = filter_empty_entities(result["entities"])
        return result["intent"], filtered_entities, result["confidence"]
    except Exception as e:
        logging.error(f"Intent detection failed: {e}")
        return "other", {}, 0.1
def filter_empty_entities(entities: dict) -> dict:
    """
    Filter out null, empty, or meaningless entities
    """
    filtered = {}
    for key, value in entities.items():
        # Skip if value is None, empty string, or meaningless values
        if value and value not in [None, "", "null", "None", "<value_or_null>", "N/A", "n/a"]:
            # Also check if it's not just whitespace
            if isinstance(value, str) and value.strip():
                filtered[key] = value.strip()
            elif not isinstance(value, str):
                filtered[key] = value
    return filtered