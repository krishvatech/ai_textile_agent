import asyncio
import logging
import os
from dateutil import parser
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import logging
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai

# ========= ENVIRONMENT SETUP =========

# ========== ENVIRONMENT SETUP ==========
load_dotenv()
api_key = os.getenv("GPT_API_KEY")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

def is_valid_rental_date(date_str: str) -> bool:
    """Returns True if rental date has day, month, and year and parses."""
    try:
        if not re.search(r"\b\d{4}\b", date_str):
            return False
        _ = parser.parse(date_str, fuzzy=True, dayfirst=True)
        return True
    except Exception:
        return False

class TextileAnalyzer:
    def __init__(self):
        # Order of info to collect
        self.entity_priority = [
            "type",      # male, female, child, unisex
            "category",  # e.g. saree, kurti, shirt
            "occasion",  # e.g. wedding, party
            "is_rental", # buy or rent
            "fabric",    # e.g. cotton, silk
            "color",     # e.g. red, blue
            "size"       # e.g. M, L, XL, Free Size
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

    def all_entity_fields_collected(self):
        "Returns True if all required initial fields are present."
        for field in self.entity_priority:
            if not self.collected_entities.get(field):
                return False
        return True

    async def fetch_variants_from_pinecone(self, entities: dict) -> List[dict]:
        """Mock function: In production, query Pinecone with entity data for best variants."""
        # This should be a true semantic or filter-based query in production!
        # Here we just return a mock list:
        return [
            {"variant_id": "1", "name": "Red Silk Saree - Free Size", "type": "female", "category": "saree", "occasion": "wedding", "color": "red", "fabric": "silk", "size": "Free Size", "is_rental": True},
            {"variant_id": "2", "name": "Blue Cotton Kurti - M", "type": "female", "category": "kurti", "occasion": "casual", "color": "blue", "fabric": "cotton", "size": "M", "is_rental": False}
        ]

    async def generate_gpt_reply(
        self,
        language: str,
        user_message: str,
        intent: str,
        filled_entities: dict,
        missing_entity: str,
        conversation_history: list,
        db_results: list = None,
        confirmation: bool = False
    ) -> str:
        lang_name = {
            "en-IN": "English",
            "hi-IN": "Hindi",
            "gu-IN": "Gujarati"
        }.get(language, "English")

        # Specific prompts for missing info
        prompt_map = {
            "type": "What kind of product do you want? (male, female, child, unisex)",
            "category": "Which product category? (saree, shirt, dress, kurti, etc)",
            "occasion": "What's the occasion for this purchase/rental? (e.g. wedding, party, office, casual)",
            "is_rental": "Do you want to buy or rent this product?",
            "fabric": "Which fabric do you want? (e.g. silk, cotton, georgette)",
            "color": "What color do you want?",
            "size": "What size do you need? (S, M, L, XL, Free Size, etc)",
            "rental_date": "Please provide the complete rental date including day, month, and year (e.g., 25 August 2025)."
        }
        if missing_entity in prompt_map:
            return prompt_map[missing_entity]

        # If showing variants, print them as menu
        if db_results is not None and len(db_results) > 0:
            lines = ["Here are some matching products:"]
            for idx, v in enumerate(db_results, 1):
                lines.append(
                    f"{idx}. {v['name']} - {v.get('fabric', '')} - {v.get('color', '')}{' (Rental available)' if v.get('is_rental') else ''}"
                )
            lines.append("Which product are you interested in? Please select by number or describe.")
            return "\n".join(lines)
        if self.asking_variant_selection and self.shown_variants:
            return "Which product are you interested in? Please select by number or describe."

        if self.asking_rental_date:
            return prompt_map["rental_date"]

        # Default
        return "Please specify your choice."

    async def analyze_message(self, text: str, tenant_id=None,language: str = "en-US") -> Dict[str, Any]:
        logging.info(f"Detected language in analyze_message: {language}")
        language=language
        logging.info(f"Detected language in analyze_message after assign: {language}")
        intent, new_entities, intent_confidence = await detect_textile_intent_openai(text, language)
        self.merge_entities(new_entities)
        self.session_history.append({"role": "user", "content": text})

        # Step 1: Greet
        if intent == "greeting" or (text.strip().lower() in [
                "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
        ]):
            self.clear_history()
            return {"input_text": text, "language": language, "detected_intent": intent, "answer": "What clothing product are you interested in today?"}

        # Step 2: Collect info fields in order
        missing_entity = self.next_missing_entity()
        if missing_entity:
            answer = await self.generate_gpt_reply(
                language, text, intent, self.collected_entities, missing_entity, self.session_history
            )
            return {
                "input_text": text,
                "language": language,
                "detected_intent": intent,
                "entities": dict(self.collected_entities),
                "intent_confidence": intent_confidence,
                "answer": answer
            }

        # Step 3: If all collected, fetch matching variants from Pinecone and show to user
        if self.shown_variants is None:
            variants = await self.fetch_variants_from_pinecone(self.collected_entities)
            if not variants:
                self.clear_history()
                return {"input_text": text, "language": language, "detected_intent": intent, "answer": "Sorry, no matching products found. Let's start again!"}
            self.shown_variants = variants
            self.asking_variant_selection = True
            answer = await self.generate_gpt_reply(
                language, text, intent, self.collected_entities, None, self.session_history, db_results=variants
            )
            return {
                "input_text": text,
                "language": language,
                "detected_intent": intent,
                "entities": dict(self.collected_entities),
                "intent_confidence": intent_confidence,
                "answer": answer
            }

        # Step 4: Await user product selection
        if self.asking_variant_selection:
            sel = self._parse_variant_selection(text)
            if sel is not None and 0 <= sel < len(self.shown_variants):
                self.selected_variant = self.shown_variants[sel]
                self.asking_variant_selection = False
                # If selected variant is rental, ask for rental_date
                if self.selected_variant.get("is_rental"):
                    self.asking_rental_date = True
                    return {
                        "input_text": text,
                        "language": language,
                        "detected_intent": intent,
                        "entities": dict(self.collected_entities),
                        "intent_confidence": intent_confidence,
                        "answer": "This product is available for rent. Please provide the complete rental date (day, month, year)."
                    }
                else:
                    # For buying, confirm
                    return {
                        "input_text": text,
                        "language": language,
                        "detected_intent": intent,
                        "entities": dict(self.collected_entities),
                        "intent_confidence": intent_confidence,
                        "answer": f"You have selected: {self.selected_variant['name']}. Please confirm to proceed with the purchase."
                    }
            else:
                # If user gave a name or partial description (fallback to LLM), you could improve this
                return {
                    "input_text": text,
                    "language": language,
                    "detected_intent": intent,
                    "entities": dict(self.collected_entities),
                    "intent_confidence": intent_confidence,
                    "answer": "Sorry, I didn't understand your selection. Please enter the product number or describe your choice."
                }

        # Step 5: Collect/validate rental date for selected variant
        if self.asking_rental_date:
            if is_valid_rental_date(text):
                self.collected_entities["rental_date"] = text
                self.asking_rental_date = False
                return {
                    "input_text": text,
                    "language": language,
                    "detected_intent": intent,
                    "entities": dict(self.collected_entities),
                    "intent_confidence": intent_confidence,
                    "answer": f"Rental confirmed for {self.selected_variant['name']} on {text}. Please confirm your order."
                }
            else:
                return {
                    "input_text": text,
                    "language": language,
                    "detected_intent": intent,
                    "entities": dict(self.collected_entities),
                    "intent_confidence": intent_confidence,
                    "answer": "Please provide the complete rental date including day, month, and year (e.g., 25 August 2025)."
                }

        # Fallback
        return {
            "input_text": text,
            "language": language,
            "detected_intent": intent,
            "entities": dict(self.collected_entities),
            "intent_confidence": intent_confidence,
            "answer": "What would you like to do next?"
        }

    def _parse_variant_selection(self, text: str) -> Optional[int]:
        """
        Try to parse a user reply as a product selection (by number). Return index or None.
        """
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
