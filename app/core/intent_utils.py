import asyncio
from openai import AsyncOpenAI
import json
import logging
import os
from typing import Tuple
from dotenv import load_dotenv
from app.core.lang_utils import detect_language # your own module assumed to be working!

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

client = AsyncOpenAI(api_key=api_key)

def process_all_entities(entities: dict) -> dict:
    """
    Return entity structure filling missing with None.
    """
    all_entities = {
        "product_name": None,
        "category": None,
        "type": None,
        "fabric": None,
        "color": None,
        "size": None,
        "price_range": None,
        "rental_price": None,
        "quantity": None,
        "location": None,
        "occasion": None,
        "is_rental": None,
        "rental_date": None
    }
    for key, value in entities.items():
        if key in all_entities:
            if value and value not in [None, "", "null", "None", "N/A", "n/a"]:
                if isinstance(value, str) and value.strip():
                    all_entities[key] = value.strip()
                elif not isinstance(value, str):
                    all_entities[key] = value
    return all_entities

async def detect_textile_intent_openai(text: str, detected_language: str) -> Tuple[str, dict, float]:
    """
    Detect customer intent and entities for textile business using OpenAI.
    Infer "type" from product category using fashion/domain norms,
    even if not explicitly mentioned in the message.
    """
    prompt = f"""
You are an AI assistant for an Indian textile business specializing in wholesale and retail.

**Your Task:** Analyze the customer message and fill as many of the below entities as possible.

**Textile Business Intents:**
1. product_search      - Looking for clothes (saree, lehenga, kurti, suit, etc.)
2. price_inquiry       - Asking about price, cost, budget
3. color_preference    - Mentioning specific colors
4. size_query          - Size, measurement, fitting questions
5. fabric_inquiry      - Cotton, silk, georgette, chiffon, etc.
6. order_placement     - Ready to buy, place order
7. order_status        - Check existing order status
8. catalog_request     - Want to see catalog, images, what's available
9. availability_check  - Stock availability
10. customization      - Tailoring, alterations, custom design
11. delivery_inquiry   - Delivery time, location, charges
12. payment_query      - Payment methods, EMI, refund
13. discount_inquiry   - Offers, deals, discounts
14. rental_inquiry     - Rental services, rental price
15. greeting           - Hello, hi, namaste
16. complaint          - Problems, issues, returns
17. other              - Anything else

**CRITICAL:**
- Always extract (and normalize) these fields:
  - **category**: Product category (saree, dress, kurti, suit, lehenga, shirt, t-shirt, frock, sherwani, etc.)
  - **type**: Always infer based only on the product category if possible, *even if user does not specify*:
    - These rules override any keyword-based guess:
      - saree, lehenga, salwar suit, anarkali, kurti, skirt, gown, frock, blouse => type: "female"
      - shirt, pant, kurta, sherwani, dhoti, pajama => type: "male"
      - "child", "kids", "boys", "girls" explicit in message => type: "child"
      - "t-shirt", "jacket", "coat", "blazer", "hoodie": type "unisex" unless user specifies gender/age group
      - If the category is ambiguous, or no clear default, leave type blank.
    - *DO NOT* require words like "ladies", "gents", "men", "women", "boys", "girls" for type – infer from category.
- If in doubt, leave blank.

**Other Entity Extraction Guidelines**:
- **product_name**: e.g. Banarasi Silk Saree, Cotton Kurti
- **fabric**: e.g. silk, cotton, georgette, etc.
- **color**: Normalize color in English.
- **size**: e.g. M, L, XL, kids, Free Size, etc.
- **price_range**: Unit price/rate per piece (NOT total amount)
- **rental_price**: Rental price if mentioned
- **quantity**: Number of pieces needed
- **location**: Delivery location
- **occasion**: Wedding, party, festival
- **is_rental**: true/false if rental inquiry
- **rental_date**: Any date/datetime phrase ("15 August", "next Tuesday", etc.)

**Customer Message:** "{text}"
**Detected Language:** "{detected_language}"

**Response Format** (JSON only):
{{
"intent": "",
"entities": {{
    "product_name": "",
    "category": "",
    "type": "",
    "fabric": "",
    "color": "",
    "size": "",
    "price_range": "",
    "rental_price": "",
    "quantity": "",
    "location": "",
    "occasion": "",
    "is_rental": "",
    "rental_date": ""
}},
"confidence": <0.0-1.0>,
"is_question": <true/false>
}}

**SAMPLE EXAMPLES (demonstrate INFERENCE):**

Message: "I want to buy a saree"
Output: {{
  "intent": "product_search",
  "entities": {{
     "product_name": "saree",
     "category": "saree",
     "type": "female"
  }},
  "confidence": 0.89,
  "is_question": false
}}

Message: "need shirt for office"
Output: {{
  "intent": "product_search",
  "entities": {{
    "product_name": "shirt",
    "category": "shirt",
    "type": "male"
  }},
  "confidence": 0.87,
  "is_question": false
}}

Message: "kurti, green cotton, free size"
Output: {{
  "intent": "product_search",
  "entities": {{
      "product_name": "kurti",
      "category": "kurti",
      "type": "female",
      "fabric": "cotton",
      "color": "green",
      "size": "Free Size"
  }},
  "confidence": 0.88,
  "is_question": false
}}

Message: "2 frocks, red and yellow"
Output: {{
  "intent": "product_search",
  "entities": {{
      "product_name": "frock",
      "category": "frock",
      "type": "female",
      "color": "red and yellow",
      "quantity": "2"
  }},
  "confidence": 0.92,
  "is_question": false
}}

Message: "sherwani for marriage"
Output: {{
  "intent": "product_search",
  "entities": {{
      "product_name": "sherwani",
      "category": "sherwani",
      "type": "male",
      "occasion": "marriage"
  }},
  "confidence": 0.85,
  "is_question": false
}}

Message: "kids t-shirt for school"
Output: {{
  "intent": "product_search",
  "entities": {{
      "product_name": "t-shirt",
      "category": "t-shirt",
      "type": "child"
  }},
  "confidence": 0.85,
  "is_question": false
}}

Message: "jacket"
Output: {{
  "intent": "product_search",
  "entities": {{
      "product_name": "jacket",
      "category": "jacket",
      "type": "unisex"
  }},
  "confidence": 0.84,
  "is_question": false
}}
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```").strip()
        result = json.loads(content)
        processed_entities = process_all_entities(result.get("entities", {}))
        return result.get("intent", "other"), processed_entities, result.get("confidence", 0.1)
    except Exception as e:
        logging.error(f"Intent detection failed: {e}")
        empty_entities = process_all_entities({})
        return "other", empty_entities, 0.1

def format_entities(entities: dict) -> str:
    if not entities:
        return "None"
    formatted_lines = []
    for k, v in entities.items():
        if v is not None and v != "":
            formatted_lines.append(f" ✓ {k}: {v}")
        else:
            formatted_lines.append(f" ○ {k}: None")
    return "\n".join(formatted_lines)

async def main():
    print("🧵 Textile Intent Detection Tester")
    print("📋 Product Schema: Banarasi Silk Saree, Cotton Kurti, etc.")
    print("Type 'q' or 'quit' to exit\n")
    while True:
        try:
            user_input = input("Enter message: ").strip()
            if user_input.lower() in ["q", "quit"]:
                print("Goodbye!")
                break
            if not user_input:
                continue
            print("🔄 Detecting language...")
            language, lang_conf = await detect_language(user_input)
            print(f"Detected Language: {language} (confidence: {lang_conf:.2f})")
            print("🔄 Detecting intent...")
            intent, entities, conf = await detect_textile_intent_openai(user_input, language)
            print("\n" + "=" * 70)
            print(f"📝 Input: {user_input}")
            print(f"🌐 Language: {language} - {lang_conf:.2f}")
            print(f"🎯 Intent: {intent} - {conf:.2f}")
            print("📋 All Entities (✓ = with value, ○ = None):")
            print(format_entities(entities))
            print("=" * 70)
            print()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")