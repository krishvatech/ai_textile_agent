import asyncio
import logging
import os
from dateutil import parser
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import logging
import re

load_dotenv()
api_key = os.getenv("GPT_API_KEY")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

def is_valid_rental_date(date_str: str) -> bool:
    try:
        if not re.search(r"\b\d{4}\b", date_str):
            return False
        _ = parser.parse(date_str, fuzzy=True, dayfirst=True)
        return True
    except Exception:
        return False

class TextileAnalyzer:
    def __init__(self):
        self.entity_priority = [
            "type", "category", "occasion", "is_rental", "fabric", "color", "size"
        ]
        self.session_history: List[Dict[str, str]] = []
        self.collected_entities: Dict[str, Any] = {}
        self.last_intent: Optional[str] = None
        self.confirmation_pending: bool = False
        self.shown_variants: Optional[List[dict]] = None
        self.selected_variant: Optional[dict] = None
        self.asking_variant_selection: bool = False
        self.asking_rental_date: bool = False
        self.gpt_client = AsyncOpenAI(api_key=api_key)
        self.logger = logging.getLogger("TextileAnalyzer")
        self.shown_variants = None
        self.asking_variant_selection = False
        
    def reset(self):
        self.session_history.clear()
        self.collected_entities.clear()
        self.last_intent = None
        self.confirmation_pending = False
        self.shown_variants = None
        self.selected_variant = None
        self.asking_variant_selection = False
        self.asking_rental_date = False
        self.shown_variants = None
        self.asking_variant_selection = False

    def clear_history(self):
        self.session_history = []
        self.collected_entities = {}
        self.last_intent = None
        self.confirmation_pending = False
        self.shown_variants = None
        self.selected_variant = None
        self.asking_variant_selection = False
        self.asking_rental_date = False

    def merge_entities(self, new_entities: dict):
        newly_gathered = {}
        for k, v in new_entities.items():
            if v and v not in ["", None, "null", "None"]:
                if self.collected_entities.get(k) != v:
                    newly_gathered[k] = v
                    self.collected_entities[k] = v
        if newly_gathered:
            gathered_str = ", ".join([f"{k}={v}" for k, v in newly_gathered.items()])
            self.logger.info(f"[INFO] New details gathered: {gathered_str}")

    def next_missing_entity(self):
        for field in self.entity_priority:
            if not self.collected_entities.get(field):
                return field
        return None

    def choose_language(self, detected_language: str) -> str:
        if detected_language.startswith("gu"):
            return "Gujarati"
        elif detected_language.startswith("hi"):
            return "Hindi"
        else:
            return "English"

    async def gpt_prompt_for_missing_entity(self, missing_entity: str, filled_entities: dict, conversation_language: str) -> str:
        lang_map = {"English": "English", "Hindi": "Hindi", "Gujarati": "Gujarati"}
        prompt = (
            f"You are a textile rental shop assistant in India. "
            f"Always reply ONLY in {lang_map.get(conversation_language, 'English')}. "
            f"Do NOT greet or thank. "
            f"Collected entity info so far (in English): {filled_entities}. "
            f"The next required info is: '{missing_entity}'. "
            f"Ask a concise, single question to collect ONLY this info from the user—in the user's language. "
            f"Do NOT ask about anything else. Just output the question."
        )
        response = await self.gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1,
            max_tokens=64
        )
        return response.choices[0].message.content.strip()

    async def fetch_variants_from_pinecone(self, entities: dict) -> List[dict]:
        return [
            {"variant_id": "1", "name": "Red Silk Saree - Free Size", "type": "female", "category": "saree", "occasion": "wedding", "color": "red", "fabric": "silk", "size": "Free Size", "is_rental": True},
            {"variant_id": "2", "name": "Blue Cotton Kurti - M", "type": "female", "category": "kurti", "occasion": "casual", "color": "blue", "fabric": "cotton", "size": "M", "is_rental": False}
        ]

    async def analyze_message(self, text: str, tenant_id=None,language: str = "en-US",intent: str | None = None,new_entities: dict | None = None,intent_confidence: float = 0.0,) -> Dict[str, Any]:
        logging.info(f"Detected language in analyze_message: {language}")
        
        if new_entities:
            self.merge_entities(new_entities)
        if intent is None:
            intent = "other"
        self.last_intent = intent 
        self.intent_confidence = intent_confidence

        self.session_history.append({"role": "user", "content": text})
        
        lang_for_prompt = self.choose_language(language)
        
        # GREETING
        if intent == "greeting" or (text.strip().lower() in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
            self.clear_history()
            greet_map = {
                "English": "What clothing product are you interested in today?",
                "Hindi": "आज आप किस कपड़े के उत्पाद में रुचि रखते हैं?",
                "Gujarati": "આજે તમે કયા કપડાની ઉત્પાદનમાં રસ રાખો છો?"
            }
            return {
                "input_text": text,
                "language": language,
                "detected_intent": intent,
                "answer": greet_map[lang_for_prompt]
            }

        # DYNAMIC GPT ENTITY QUESTION
        missing_entity = self.next_missing_entity()
        if missing_entity:
            answer = await self.gpt_prompt_for_missing_entity(
                missing_entity, self.collected_entities, lang_for_prompt
            )
            return {
                "input_text": text,
                "language": language,
                "detected_intent": intent,
                "entities": dict(self.collected_entities),
                "intent_confidence": intent_confidence,
                "answer": answer
            }

        # Show variants
        if self.shown_variants is None:
            variants = await self.fetch_variants_from_pinecone(self.collected_entities)
            if not variants:
                self.clear_history()
                sorry_map = {
                    "English": "Sorry, no matching products found. Let's start again!",
                    "Hindi": "क्षमा करें, कोई उपयुक्त उत्पाद नहीं मिला। चलिए फिर से शुरू करते हैं!",
                    "Gujarati": "માફ કરો, મેળ ખાતા ઉત્પાદનો મળ્યા નથી. ફરી શરૂ કરીએ!"
                }
                return {
                    "input_text": text,
                    "language": language,
                    "detected_intent": intent,
                    "answer": sorry_map[lang_for_prompt]
                }
            self.shown_variants = variants
            self.asking_variant_selection = True
            lines = {
                "English": ["Here are some matching products:"],
                "Hindi": ["यहाँ कुछ उपयुक्त उत्पाद हैं:"],
                "Gujarati": ["અહીં કેટલીક મેળ ખાતી ઉત્પાદનો છે:"]
            }[lang_for_prompt]
            for idx, v in enumerate(variants, 1):
                lines.append(
                    f"{idx}. {v['name']} - {v.get('fabric', '')} - {v.get('color', '')}{' (Rental available)' if v.get('is_rental') else ''}"
                )
            select_map = {
                "English": "Which product are you interested in? Please select by number or describe.",
                "Hindi": "आप कौन सा उत्पाद चुनना चाहते हैं? कृपया नंबर से या विवरण से चुनें।",
                "Gujarati": "તમે કયું ઉત્પાદન પસંદ કરવું છે? કૃપા કરીને નંબર કે વર્ણનથી પસંદ કરો."
            }
            lines.append(select_map[lang_for_prompt])
            answer = "\n".join(lines)
            return {
                "input_text": text,
                "language": language,
                "detected_intent": intent,
                "entities": dict(self.collected_entities),
                "intent_confidence": intent_confidence,
                "answer": answer
            }

        # Product selection
        if self.asking_variant_selection:
            sel = self._parse_variant_selection(text)
            if sel is not None and 0 <= sel < len(self.shown_variants):
                self.selected_variant = self.shown_variants[sel]
                self.asking_variant_selection = False
                if self.selected_variant.get("is_rental"):
                    self.asking_rental_date = True
                    reply_map = {
                        "English": "This product is available for rent. Please provide the complete rental date (day, month, year).",
                        "Hindi": "यह उत्पाद किराए के लिए उपलब्ध है। कृपया पूरी तारीख (दिन, महीना, वर्ष) बताएं।",
                        "Gujarati": "આ ઉત્પાદન ભાડે માટે ઉપલબ્ધ છે. કૃપા કરીને સંપૂર્ણ ભાડાની તારીખ (દિન, મહિનો, વર્ષ) આપો."
                    }
                    reply = reply_map[lang_for_prompt]
                    return {
                        "input_text": text,
                        "language": language,
                        "detected_intent": intent,
                        "entities": dict(self.collected_entities),
                        "intent_confidence": intent_confidence,
                        "answer": reply
                    }
                else:
                    confirm_map = {
                        "English": f"You have selected: {self.selected_variant['name']}. Please confirm to proceed with the purchase.",
                        "Hindi": f"आपने चुना है: {self.selected_variant['name']}. कृपया खरीदारी आगे बढ़ाने के लिए पुष्टि करें।",
                        "Gujarati": f"તમે પસંદ કર્યું છે: {self.selected_variant['name']}. ખરીદી પૂરું પાડવા માટે કૃપા કરીને પુષ્ટિ કરો."
                    }
                    return {
                        "input_text": text,
                        "language": language,
                        "detected_intent": intent,
                        "entities": dict(self.collected_entities),
                        "intent_confidence": intent_confidence,
                        "answer": confirm_map[lang_for_prompt]
                    }
            else:
                sorry_map = {
                    "English": "Sorry, I didn't understand your selection. Please enter the product number or describe your choice.",
                    "Hindi": "माफ़ कीजिए, मैं आपका चयन नहीं समझ पाया। कृपया उत्पाद का नंबर या विवरण दर्ज करें।",
                    "Gujarati": "માફ કરો, મેં તમારો પસંદગીઓ સમજ્યો નથી. કૃપા કરીને પસંદગીઓનો નંબર કે વર્ણન લખો."
                }
                return {
                    "input_text": text,
                    "language": language,
                    "detected_intent": intent,
                    "entities": dict(self.collected_entities),
                    "intent_confidence": intent_confidence,
                    "answer": sorry_map[lang_for_prompt]
                }

        # Rental date
        if self.asking_rental_date:
            if is_valid_rental_date(text):
                self.collected_entities["rental_date"] = text
                self.asking_rental_date = False
                confirm_map = {
                    "English": f"Rental confirmed for {self.selected_variant['name']} on {text}. Please confirm your order.",
                    "Hindi": f"{self.selected_variant['name']} का किराया {text} के लिए पक्का। कृपया अपना ऑर्डर कन्फर्म करें।",
                    "Gujarati": f"{self.selected_variant['name']}નું ભાડું {text} માટે પુષ્ટિ થયેલ છે. કૃપા કરીને ઓર્ડર કનફર્મ કરો."
                }
                reply = confirm_map[lang_for_prompt]
                return {
                    "input_text": text,
                    "language": language,
                    "detected_intent": intent,
                    "entities": dict(self.collected_entities),
                    "intent_confidence": intent_confidence,
                    "answer": reply
                }
            else:
                answer = await self.gpt_prompt_for_missing_entity(
                    "rental_date", self.collected_entities, lang_for_prompt
                )
                return {
                    "input_text": text,
                    "language": language,
                    "detected_intent": intent,
                    "entities": dict(self.collected_entities),
                    "intent_confidence": intent_confidence,
                    "answer": answer
                }
        fallback_map = {
            "English": "What would you like to do next?",
            "Hindi": "अब आप क्या करना चाहते हैं?",
            "Gujarati": "હવે તમે શું કરવું ઇચ્છો છો?"
        }
        return {
            "input_text": text,
            "language": language,
            "detected_intent": intent,
            "entities": dict(self.collected_entities),
            "intent_confidence": intent_confidence,
            "answer": fallback_map[lang_for_prompt]
        }

    def _parse_variant_selection(self, text: str) -> Optional[int]:
        match = re.match(r"\D*(\d+)", text.strip())
        if match:
            idx = int(match.group(1)) - 1
            return idx
        return None

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

