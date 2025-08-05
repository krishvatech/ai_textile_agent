import asyncio
import json
import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai
from openai import AsyncOpenAI

load_dotenv()

api_key = os.getenv("GPT_API_KEY")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

client = AsyncOpenAI(api_key=api_key)

class TextileAnalyzer:
    def __init__(self):
        self.supported_languages = ["hi-IN", "gu-IN", "en-IN"]
        self.session_history = []
        self.language_history = []
        self.session_language = None
        self.collected_entities = {}
        self.message_count = 0
        self.questions_asked = set()
        self.conversation_stage = "initial"
        self.rent_checked = False

        # Strict entity order (this is what your bot will follow for product_search)
        self.entity_priority = {
            "product_search": [
                "is_rental",   # Buy or rent
                "occasion",    # After buy/rent, ask for occasion
                "fabric",      # Then ask for fabric
                "size",        # Then ask for size
                "color",       # Then ask for color
                "rental_date"  # Rental date last (only for rent)
            ]
        }

    def clear_history(self):
        self.session_history = []
        self.language_history = []
        self.session_language = None
        self.collected_entities = {}
        self.message_count = 0
        self.questions_asked = set()
        self.conversation_stage = "initial"
        self.rent_checked = False

    def merge_entities(self, new_entities: dict):
        for k, v in new_entities.items():
            if v and v not in ["", None, "null", "None"]:
                self.collected_entities[k] = v

    def get_missing_priority_entities(self, intent: str) -> list[str]:
        priorities = self.entity_priority.get(intent, [])
        missing = []
        for entity in priorities:
            if entity == "rental_date":
                if self.collected_entities.get("is_rental") in [True, "true", "True"]:
                    if not self.collected_entities.get(entity):
                        missing.append(entity)
            else:
                if not self.collected_entities.get(entity):
                    missing.append(entity)
        return missing[:1]  # One entity at a time

    def update_conversation_stage(self, intent: str):
        priorities = self.entity_priority.get(intent, [])
        count = sum(1 for p in priorities if self.collected_entities.get(p))
        if count == 0:
            self.conversation_stage = "initial"
        elif count < len(priorities) * 0.6:
            self.conversation_stage = "gathering_details"
        else:
            self.conversation_stage = "finalizing"

    async def check_rental_availability(self, category, rental_date):
        # Simulate DB check (replace with your pgAdmin/DB code as needed)
        unavailable = ['2025-08-31', '31st August']
        return str(rental_date).strip().lower() not in [d.lower() for d in unavailable]

    async def create_rental_order(self, user_info, entities):
        # Simulate booking logic
        return True

    # ----- THIS IS THE MAIN FLOW LOGIC -----
    async def generate_contextual_reply(self, language, intent, entities, user_text, missing_entities) -> str:
        lang_map = {"hi-IN": "Hindi", "gu-IN": "Gujarati", "en-IN": "English"}
        greetings = {
            "hi-IN": [
                "नमस्ते! आपको कैसी मदद चाहिए? आप कपड़े देखना चाहते हैं?",
                "स्वागत है! आपकी क्या मदद कर सकती हूँ?",
            ],
            "gu-IN": [
                "નમસ્તે! તમને કેવી મદદ જોઈએ? તમે કપડાં જોવા માંગો છો?",
                "સ્વાગત છે! હું શું સહાય કરી શકું?",
            ],
            "en-IN": [
                "Hello! How can I help you today? Are you looking for any clothes?",
                "Welcome! How may I assist you?",
            ]
        }
        # Normalize user_text for greeting recognition
        cleansed = user_text.lower().strip()
        greeting_words = [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "नमस्ते", "હાય", "હેલો", "હાય", "hello there"
        ]
        # Compose greeting logic
        if (
            intent == "greeting"
            or any(g in cleansed for g in greeting_words)
        ):
            # Pick a random greeting message in the selected language
            import random
            greet_list = greetings.get(language, greetings["en-IN"])
            return random.choice(greet_list)
        q = {
            "hi-IN": {
                "is_rental": "क्या आप {} खरीदना चाहते हैं या किराए पर लेना चाहते हैं?",
                "occasion": "यह किस अवसर के लिए है?",
                "fabric": "कौन सा कपड़ा पसंद है? (सिल्क, कॉटन, जॉर्जेट)",
                "size": "आपको कौन सा साइज चाहिए? (S, M, L, XL या Free Size)",
                "color": "कौन सा रंग पसंद है? (लाल, नीला, हरा...)",
                "rental_date": "किराए की तारीख क्या है? किस दिन चाहिए?",
            },
            "gu-IN": {
                "is_rental": "તમે {} ખરીદવા માંગો છો કે ભાડે લેવા માંગો છો?",
                "occasion": "આ કયા પ્રસંગ માટે છે?",
                "fabric": "કયો કાપડ પસંદ છે? (સિલ્ક, કોટન, જ્યોર્જેટ)",
                "size": "કયો સાઇઝ જોઇએ? (S, M, L, XL કે Free Size)",
                "color": "કયો કલર પસંદ છે? (લાલ, નીલો, લીલો...)", 
                "rental_date": "કયા દિવસે ભાડે જોઇએ?",
            },
            "en-IN": {
                "is_rental": "Do you want to buy or rent this product?",
                "occasion": "What's the occasion?",
                "fabric": "Which fabric do you prefer? (Silk, Cotton, Georgette)",
                "size": "What size do you need? (S, M, L, XL or Free Size)",
                "color": "Which color would you like? (red, blue, green...)", 
                "rental_date": "What's your rental date? (When do you need it?)",
            }
        }
        templates = q.get(language, q["en-IN"])

        # Build the question-order
        strict_order = self.entity_priority.get("product_search", [])

        # Build which field is missing, in order
        ordered_missing = []
        for e in strict_order:
            if e == "rental_date":
                if entities.get("is_rental") in [True, "true", "True"] and not entities.get(e):
                    ordered_missing.append(e)
            else:
                if not entities.get(e):
                    ordered_missing.append(e)

        # Always ask for exactly one entity, in your desired order
        if ordered_missing:
            next_entity = ordered_missing[0]
            self.questions_asked.add(next_entity)
            # For is_rental, format with category if available
            if next_entity == "is_rental" and 'category' in entities and '{}' in templates["is_rental"]:
                return templates["is_rental"].format(entities['category'])
            return templates.get(next_entity, f"Please specify {next_entity}")

        # If renting, after rental_date, check DB and only proceed if available
        if entities.get("is_rental") in [True, "true", "True"] and entities.get("rental_date") and not self.rent_checked:
            category = entities.get("category", "product")
            rental_date = entities.get("rental_date")
            is_available = await self.check_rental_availability(category, rental_date)
            self.rent_checked = True
            if is_available:
                return f"Yes, {category} is available on {rental_date}. Please confirm to proceed with the booking."
            else:
                self.collected_entities["rental_date"] = None
                self.questions_asked.discard("rental_date")
                self.rent_checked = False
                return f"Sorry, {category} is not available on {rental_date}. Please provide another rental date."

        # Show summary/confirmation ONLY when all fields in desired order are filled
        collected_summary = []
        for e in strict_order:
            if e == "rental_date" and not (entities.get("is_rental") in [True, "true", "True"]):
                continue
            if entities.get(e):
                collected_summary.append(f"{e}: {entities.get(e)}")
        summary_str = ', '.join(collected_summary)
        return f"Review your request: {summary_str}. Confirm to proceed."

    async def generate_ai_reply(self, language: str, intent: str, entities: dict, user_text: str, missing_entities: list[str]) -> str:
        lang_map = {"hi-IN": "Hindi", "gu-IN": "Gujarati", "en-IN": "English"}
        language_name = lang_map.get(language, "English")
        prompt = f"""
Context: Textile shop assistant conversation
User's intent: {intent}
Collected info: {json.dumps(self.collected_entities, ensure_ascii=False)}
Missing priority info: {missing_entities}
Conversation stage: {self.conversation_stage}
Questions already asked: {list(self.questions_asked)}
Generate a relevant, non-repetitive follow-up in {language_name} script.
"""
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Reply only in {language_name} script. Be warm and helpful."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"AI reply failed: {e}")
            return "Please tell me more."

    async def analyze_message(self, text: str) -> Dict[str, Any]:
        try:
            language, lang_confidence = await detect_language(text)
            self.message_count += 1
            previous_entities = dict(self.collected_entities)
            # Lock session language after N messages
            N = 3
            if self.session_language is None:
                self.language_history.append(language)
                if len(self.language_history) == N:
                    from collections import Counter
                    lang_counts = Counter(self.language_history)
                    self.session_language = lang_counts.most_common(1)[0][0]
                detected_language = language
            else:
                detected_language = self.session_language
            intent, new_entities, intent_confidence = await detect_textile_intent_openai(text, detected_language)
            self.merge_entities(new_entities)
            self.update_conversation_stage(intent)
            self.session_history.append({"role": "user", "content": text})
            newly_added = {}
            for k, v in self.collected_entities.items():
                if k not in previous_entities and v:
                    newly_added[k] = v
            response_text = None
            # Confirm/Book
            if "confirm" in text.lower() or "book" in text.lower():
                order_ok = await self.create_rental_order(None, self.collected_entities)
                if order_ok:
                    response_text = (
                        "Booking confirmed! You'll receive a WhatsApp message with details. "
                        "You can pay at the store or online. Need anything else?"
                    )
            missing_entities = self.get_missing_priority_entities(intent)
            if not response_text:
                response_text = await self.generate_contextual_reply(
                    detected_language, intent, self.collected_entities, text, missing_entities
                )
            self.session_history.append({"role": "assistant", "content": response_text})
            return {
                "input_text": text,
                "detected_language": detected_language,
                "language_confidence": lang_confidence,
                "detected_intent": intent,
                "intent_confidence": intent_confidence,
                "entities": dict(self.collected_entities),
                "newly_added_entities": newly_added,
                "missing_entities": missing_entities,
                "conversation_stage": self.conversation_stage,
                "answer": response_text,
            }
        except Exception as e:
            logging.error(f"Analysis pipeline failed: {e}")
            return {
                "input_text": text,
                "detected_language": self.session_language or "en-IN",
                "language_confidence": 0.1,
                "detected_intent": "other",
                "intent_confidence": 0.1,
                "entities": {},
                "newly_added_entities": {},
                "missing_entities": [],
                "answer": "Sorry, something went wrong.",
                "error": str(e)
            }