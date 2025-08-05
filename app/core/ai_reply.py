import asyncio
import logging
import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai

# ========= ENVIRONMENT SETUP =========

# ========== ENVIRONMENT SETUP ==========
load_dotenv()
api_key = os.getenv("GPT_API_KEY")
if not api_key:
    print("❌ Error: GPT_API_KEY not found in environment variables")
    exit(1)

class TextileAnalyzer:
    def __init__(self):
        self.entity_priority = {
            "product_search": [
                "is_rental", "occasion", "fabric", "size", "color", "rental_date"
            ]
        }
        self.session_history: List[Dict[str, str]] = []
        self.collected_entities: Dict[str, Any] = {}
        self.last_intent: Optional[str] = None
        self.confirmation_pending: bool = False
        self.gpt_client = AsyncOpenAI(api_key=api_key)
        self.logger = logging.getLogger("TextileAnalyzer")

    def clear_history(self):
        self.session_history = []
        self.collected_entities = {}
        self.last_intent = None
        self.confirmation_pending = False

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

    def has_product_search_entities(self, entity_dict):
        product_fields = {"category", "product_name", "fabric", "size", "color", "occasion", "is_rental", "rental_date"}
        return any(entity_dict.get(field) for field in product_fields)

    def next_missing_entity(self, intent: str):
        for field in self.entity_priority.get(intent, []):
            if not self.collected_entities.get(field):
                return field
        return None

    async def fetch_rental_products(self, tenant_id, occasion, fabric, size, color, rental_date):
        details_log = f"is_rental={self.collected_entities.get('is_rental')}, occasion={occasion}, fabric={fabric}, size={size}, color={color}, rental_date={rental_date}"
        self.logger.info(f"[INFO] Finding in database with: {details_log}")
        print(f"Finding in database with: {details_log}")
        # Mock DB logic
        if all([occasion, fabric, size, color, rental_date]):
            return [
                {"product_name": "Red Silk Saree", "available": True},
                {"product_name": "Blue Cotton Kurti", "available": True}
            ]
        else:
            return []

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
        system_prompt = (
            f"You are a textile assistant for a shop in India. "
            f"Always reply ONLY in {lang_name} script—never use another language or script. "
            f"Do NOT greet or thank the user; never reply with 'Hello', 'Thank you', or similar. "
            f"Strictly avoid repetition. Always be concise and get to the next question, missing info, or result summary only."
            f"If you see a date-like phrase, treat it as rental_date and never treat it as occasion if a date phrase like '15 august', '30 august', '31/8/25', 'next Tuesday', etc. appears."
        )
        user_prompt = f"""
User message: {user_message}
Detected intent: {intent}
Entities so far: {filled_entities}
Next missing entity (if any): {missing_entity}
Conversation history: {conversation_history}
Database results: {db_results if db_results is not None else 'N/A'}
Instructions:
- If the intent is greeting, immediately ask what product/service the user needs, but DO NOT greet or thank them.
- If the intent is product_search AND some info is missing, just ask a single, specific, non-repetitive question for the next missing entity. DO NOT greet, thank, or restate context.
- If all rental fields are filled and confirmation is not yet given, present ALL the provided details in a clear, formatted summary (e.g., 'You want to rent a red cotton saree, size free-size, for a wedding on 15 August. Please confirm (yes/no).'). Wait for user to confirm before searching the database.
- If confirmation has been given, show results (from DB) or ask to proceed.
- For all other cases, respond with the next logical actionable question.
- Be brief, never repeat information or entities already in the conversation.
- Never add introductory phrases, greetings, closings, or 'thank yous'.
"""
        if confirmation:
            user_prompt += "\nThe user has confirmed the details, now proceed to show rental product options based on these details."
        response = await self.gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=160,
        )
        return response.choices[0].message.content.strip()

    async def gpt_confirms(self, language: str, user_message: str, summary: str) -> str:
        lang_name = {
            "en-IN": "English", "hi-IN": "Hindi", "gu-IN": "Gujarati"
        }.get(language, "English")
        system_prompt = (
            f"You are a polite AI for a shop in India. "
            f"Your answer must be ONLY one word: confirm, reject, or neither. Strictly no extra text."
            f"If the user says a date or time or extra rental date, do NOT consider it a confirmation, only update rental_date on the backend."
        )
        user_prompt = f"""
User message: {user_message}
Order summary: {summary}
Rules:
- Reply 'confirm' if user is CLEARLY accepting, confirming, or agreeing to proceed (including 'yes', 'proceed', 'that's correct', etc.).
- Reply 'reject' if user is declining, canceling, or wants to modify any details.
- Reply 'neither' if message is unclear, not a confirmation or a rejection, or if the only thing the user says is a new value for one of the fields (e.g., 'I want on 30 August' means update the rental_date, not confirmation).
Only reply: confirm, reject, or neither. No extra words.
"""
        response = await self.gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip().lower()

    async def analyze_message(self, text: str, tenant_id=None) -> Dict[str, Any]:
        language, _ = await detect_language(text)
        intent, new_entities, intent_confidence = await detect_textile_intent_openai(text, language)
        # Detect if message is product-related even if intent not clear
        product_related = self.has_product_search_entities(new_entities)

        # Intent state-machine management
        active_intent = (
            "product_search" if intent == "product_search" or product_related
            else "greeting" if intent == "greeting"
            else "product_search" if self.last_intent == "product_search"
            else "greeting"
        )

        self.merge_entities(new_entities)
        self.session_history.append({"role": "user", "content": text})

        if active_intent == "product_search":
            self.last_intent = "product_search"
        elif active_intent == "greeting":
            self.last_intent = "greeting"

        filled = {k: v for k, v in self.collected_entities.items() if v}
        missing_entity = self.next_missing_entity("product_search") if active_intent == "product_search" else None

        # Step 1: Ask for next info until all required
        if (
            active_intent == "product_search"
            and missing_entity is None
            and not self.confirmation_pending
        ):
            self.confirmation_pending = True
            self.logger.info(f"[INFO] All details gathered: {filled}")
            answer = await self.generate_gpt_reply(
                language=language,
                user_message=text,
                intent=active_intent,
                filled_entities=filled,
                missing_entity=missing_entity,
                conversation_history=self.session_history,
                db_results=None,
                confirmation=False
            )
            return {
                "input_text": text,
                "language": language,
                "detected_intent": intent,
                "active_intent": active_intent,
                "entities": dict(self.collected_entities),
                "intent_confidence": intent_confidence,
                "answer": answer
            }

        # Step 2: Confirmation dialog and entity update if missing/corrected
        if self.confirmation_pending:
            summary = ", ".join([f"{k}={v}" for k, v in filled.items()])
            gpt_decision = await self.gpt_confirms(language, text, summary)
            if gpt_decision == "confirm":
                self.confirmation_pending = False
                db_results = await self.fetch_rental_products(
                    tenant_id=tenant_id,
                    occasion=self.collected_entities.get("occasion"),
                    fabric=self.collected_entities.get("fabric"),
                    size=self.collected_entities.get("size"),
                    color=self.collected_entities.get("color"),
                    rental_date=self.collected_entities.get("rental_date")
                )
                answer = await self.generate_gpt_reply(
                    language=language,
                    user_message=text,
                    intent=active_intent,
                    filled_entities=filled,
                    missing_entity=None,
                    conversation_history=self.session_history,
                    db_results=db_results,
                    confirmation=True
                )
                return {
                    "input_text": text,
                    "language": language,
                    "detected_intent": intent,
                    "active_intent": active_intent,
                    "entities": dict(self.collected_entities),
                    "intent_confidence": intent_confidence,
                    "answer": answer
                }
            elif gpt_decision == "reject":
                self.logger.info("[INFO] User rejected or wants to change details. Restarting.")
                self.clear_history()
                return {
                    "input_text": text,
                    "language": language,
                    "detected_intent": intent,
                    "active_intent": "greeting",
                    "entities": dict(), # reset
                    "intent_confidence": intent_confidence,
                    "answer": "Order canceled or details reset. Let's start again. What do you want to rent/buy?"
                }
            else:
                # If the user supplied a new value (e.g., a new rental_date or field update), always merge—then recheck missing/confirmation
                intent2, new_entities2, _ = await detect_textile_intent_openai(text, language)
                self.merge_entities(new_entities2)
                filled2 = {k: v for k, v in self.collected_entities.items() if v}
                missing_entity2 = self.next_missing_entity("product_search")
                if missing_entity2 is None:
                    # All filled after update—re-ask confirm
                    summary2 = ", ".join([f"{k}={v}" for k, v in filled2.items()])
                    answer = await self.generate_gpt_reply(
                        language=language,
                        user_message=text,
                        intent=active_intent,
                        filled_entities=filled2,
                        missing_entity=None,
                        conversation_history=self.session_history,
                        db_results=None,
                        confirmation=False
                    )
                    return {
                        "input_text": text,
                        "language": language,
                        "detected_intent": intent2,
                        "active_intent": active_intent,
                        "entities": dict(self.collected_entities),
                        "intent_confidence": intent_confidence,
                        "answer": answer
                    }
                else:
                    # There is still some missing entity after user update
                    answer = await self.generate_gpt_reply(
                        language=language,
                        user_message=text,
                        intent=active_intent,
                        filled_entities=filled2,
                        missing_entity=missing_entity2,
                        conversation_history=self.session_history,
                        db_results=None,
                        confirmation=False
                    )
                    return {
                        "input_text": text,
                        "language": language,
                        "detected_intent": intent2,
                        "active_intent": active_intent,
                        "entities": dict(self.collected_entities),
                        "intent_confidence": intent_confidence,
                        "answer": answer
                    }

        # Step 3: Default/fallback GPT reply
        answer = await self.generate_gpt_reply(
            language=language,
            user_message=text,
            intent=active_intent,
            filled_entities=filled,
            missing_entity=missing_entity,
            conversation_history=self.session_history,
            db_results=None,
            confirmation=False
        )
        return {
            "input_text": text,
            "language": language,
            "detected_intent": intent,
            "active_intent": active_intent,
            "entities": dict(self.collected_entities),
            "intent_confidence": intent_confidence,
            "answer": answer
        }

    # CLI for dev/demo/testing
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

# ============= MAIN ===============
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = TextileAnalyzer()
    asyncio.run(analyzer.run_cli())