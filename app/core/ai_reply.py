import asyncio
import logging
import os
from typing import Dict, Any

from dotenv import load_dotenv
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai

# ========= ENVIRONMENT SETUP =========

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

class TextileAnalyzer:
    def __init__(self):
        self.entity_priority = {
            "product_search": [
                "is_rental",
                "occasion",
                "fabric",
                "size",
                "color",
                "rental_date"
            ]
        }
        self.greetings = {
            "en-IN": "Hello! Welcome to our textile rental service. What product or service would you like to rent today?",
            "hi-IN": "नमस्ते! हमारे कपड़ा किराया सेवा में आपका स्वागत है। आप किस वस्त्र या सेवा को किराए पर लेना चाहते हैं?",
            "gu-IN": "નમસ્તે! અમારી ટેક્સટાઇલ ભાડે સેવા માં આપનું સ્વાગત છે. તમે કયો માલ અથવા સેવા ભાડે પર લઈ શોધી રહ્યા છો?"
        }
        # Multilingual rental questions
        self.rent_questions_map = {
            "en-IN": {
                "occasion": "What is the occasion?",
                "fabric": "Which fabric do you prefer? (Silk, Cotton, Georgette, etc.)",
                "size": "What size do you need? (S, M, L, XL, or Free Size)",
                "color": "What color would you like?",
                "rental_date": "Please tell us the rental date."
            },
            "hi-IN": {
                "occasion": "यह किस अवसर के लिए है?",
                "fabric": "आप किस कपड़े को पसंद करते हैं? (सिल्क, कॉटन, जॉर्जेट आदि)",
                "size": "आपको कौन सा साइज चाहिए? (S, M, L, XL या फ्री साइज)",
                "color": "आपको कौन सा रंग चाहिए?",
                "rental_date": "किराए की तारीख क्या है?"
            },
            "gu-IN": {
                "occasion": "આ કયા અવસરે માટે છે?",
                "fabric": "તમને કયો કાપડ જોઈએ? (સિલ્ક, કોટન, જ્યોર્જેટ વગેરે)",
                "size": "તમને કયો સાઈઝ જોઈએ? (S, M, L, XL કે ફ્રી સાઈઝ)",
                "color": "તમને કયો કલર જોઈએ?",
                "rental_date": "ભાડે માટે કયારે જોઈએ છે?"
            }
        }
        self.rent_results_text = {
            "en-IN": "Here are available rental products based on your preferences: ",
            "hi-IN": "आपकी पसंद के अनुसार ये किराए पर उपलब्ध वस्त्र हैं: ",
            "gu-IN": "તમારી પસંદગીને આધારે હાજર ભાડે માલ: "
        }
        self.rent_no_results_text = {
            "en-IN": "Sorry, no products match your rental criteria. Please modify your choices.",
            "hi-IN": "माफ करें, आपके विकल्पों के अनुसार कोई वस्त्र उपलब्ध नहीं है। कृपया अपनी पसंद बदलें।",
            "gu-IN": "માફ કરો, તમારી પસંદગીઓ પર ખાતરીબદ્ધ વસ્તુઓ મળ્યા નથી. કૃપા કરીને પસંદ બદલો."
        }
        self.session_history = []
        self.collected_entities = {}
        self.questions_asked = set()
        self.conversation_stage = "initial"
        self.last_intent = None

    def clear_history(self):
        self.session_history = []
        self.collected_entities = {}
        self.questions_asked = set()
        self.conversation_stage = "initial"
        self.last_intent = None

    def merge_entities(self, new_entities: dict):
        for k, v in new_entities.items():
            if v and v not in ["", None, "null", "None"]:
                self.collected_entities[k] = v

    def get_missing_priority_entities(self, intent: str):
        priorities = self.entity_priority.get(intent, [])
        return [e for e in priorities if not self.collected_entities.get(e)]

    def has_product_search_entities(self, entity_dict):
        product_fields = {"category", "product_name", "fabric", "size", "color", "occasion", "is_rental", "rental_date"}
        return any(entity_dict.get(field) for field in product_fields)

    async def fetch_rental_products(self, tenant_id, occasion, fabric, size, color, rental_date):
        # TODO: Replace with real DB/API logic!
        print(f"Fetching rental products for: tenant_id={tenant_id}, occasion={occasion}, fabric={fabric}, size={size}, color={color}, rental_date={rental_date}")
        return [
            {"product_name": "Red Silk Saree", "available": True},
            {"product_name": "Blue Cotton Kurti", "available": True}
        ]

    async def handle_buy_flow(self, entities):
        return "Buy flow is not implemented yet."

    async def rent_flow(self, tenant_id, language):
        """Ask missing rental fields or fetch from DB if all present."""
        q_map = self.rent_questions_map.get(language, self.rent_questions_map["en-IN"])
        required_fields = list(q_map.keys())  # Keeps order consistent

        for field in required_fields:
            if not self.collected_entities.get(field):
                return q_map[field]

        results = await self.fetch_rental_products(
            tenant_id=tenant_id,
            occasion=self.collected_entities.get("occasion"),
            fabric=self.collected_entities.get("fabric"),
            size=self.collected_entities.get("size"),
            color=self.collected_entities.get("color"),
            rental_date=self.collected_entities.get("rental_date")
        )
        if results:
            resp_text = self.rent_results_text.get(language, self.rent_results_text["en-IN"])
            resp = resp_text + ", ".join([x["product_name"] for x in results])
        else:
            resp = self.rent_no_results_text.get(language, self.rent_no_results_text["en-IN"])
        return resp

    async def analyze_message(self, text: str, tenant_id=None) -> Dict[str, Any]:
        # Step 1: Detect language
        
        language, lang_confidence = await detect_language(text)
        # Step 2: Detect intent/entities
        intent, new_entities, intent_confidence = await detect_textile_intent_openai(text, language)
        # Step 3: Promote intent if new product-related info
        product_related = self.has_product_search_entities(new_entities)
        if intent == "product_search" or product_related:
            active_intent = "product_search"
        elif intent == "greeting":
            active_intent = "greeting"
        elif self.last_intent == "product_search":
            active_intent = "product_search"
        else:
            active_intent = "greeting"

        # Step 4: Always merge entities
        self.merge_entities(new_entities)
        self.session_history.append({"role": "user", "content": text})

        # Step 5: Save last actionable intent
        if active_intent == "product_search":
            self.last_intent = "product_search"
        elif active_intent == "greeting":
            self.last_intent = "greeting"

        # Step 6: Main logic branch using detected language
        if active_intent == "greeting":
            
            resp = self.greetings.get(language, self.greetings["en-IN"])
        elif active_intent == "product_search":
            is_rental = self.collected_entities.get("is_rental")
            if is_rental not in [True, "true", "True", False, "false", "False"]:
                # Multilingual rent/buy question
                choice_ask = {
                    "en-IN": "Do you want to buy or rent the product? (Currently, only rent is supported.)",
                    "hi-IN": "क्या आप उत्पाद खरीदना चाहते हैं या किराए पर लेना चाहते हैं? (फिलहाल, केवल किराया उपलब्ध है।)",
                    "gu-IN": "તમે ઉત્પાદન ખરીદવા માંગો છો કે ભાડે લેવા માંગો છો? (હમણાં, માત્ર ભાડે છે.)"
                }
                resp = choice_ask.get(language, choice_ask["en-IN"])
            elif str(is_rental).lower() in ["false", "no"]:
                resp = await self.handle_buy_flow(self.collected_entities)
            else:
                resp = await self.rent_flow(tenant_id, language)
        else:
            sorry = {
                "en-IN": "Sorry, I couldn't understand your request. Please try again!",
                "hi-IN": "माफ़ कीजिए, मैं आपका अनुरोध समझ नहीं सका। कृपया फिर से प्रयास करें!",
                "gu-IN": "માફ કરો, હું તમારો વિનંતી સમજ્યો નહિ. કૃપા કરીને ફરી પ્રયાસ કરો!"
            }
            resp = sorry.get(language, sorry["en-IN"])

        return {
            "input_text": text,
            "language": language,
            "detected_intent": intent,
            "active_intent": active_intent,
            "entities": dict(self.collected_entities),
            "intent_confidence": intent_confidence,
            "answer": resp
        }

    async def run_cli(self):
        print("Welcome to Textile Rental Assistant!")
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["q", "quit"]:
                    print("Assistant: Goodbye!")
                    break
                result = await self.analyze_message(user_input, tenant_id="your-tenant-id")
                print(f"Assistant: {result['answer']}")
            except KeyboardInterrupt:
                print("\nSession ended. Bye!")
                break









