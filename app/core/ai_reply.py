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
        self.language_history = []  # List of first N detected user languages
        self.session_language = None  # Will be locked after N messages
        self.collected_entities = {}
        self.message_count = 0

    def clear_history(self):
        self.session_history = []
        self.language_history = []
        self.session_language = None
        self.collected_entities = {}
        self.message_count = 0

    def merge_entities(self, new_entities: dict):
        for k, v in new_entities.items():
            if v and v not in ["", None, "null"]:
                self.collected_entities[k] = v

    async def generate_reply(
        self,
        language: str,
        intent: str,
        entities: dict,
        user_text: str,
        clarification_needed: bool = False
    ) -> str:
        # Setup logger
        logger = logging.getLogger("TextileAssistant")
        # Set logging level if not already set
        if not logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)

        lang_map = {"hi-IN": "Hindi", "gu-IN": "Gujarati", "en-IN": "English"}
        language_name = lang_map.get(language, "English")

        # ---- NEW LOGIC FOR PRODUCT_SEARCH ----
        if intent == "product_search":
            product_name = entities.get("product", "product")
            # If rental status not known, ask the question
            if "is_rental" not in entities:
                question_map = {
                    "hi-IN": f"क्या आप {product_name} खरीदना चाहते हैं या किराए पर लेना चाहते हैं?",
                    "gu-IN": f"તમે {product_name} ખરીદવા માંગો છો કે ભાડે લેવા માંગો છો?",
                    "en-IN": f"Are you looking to buy or rent for this {product_name}?"
                }
                logger.info(f"Asked user: {question_map.get(language, question_map['en-IN'])}")
                return question_map.get(language, question_map["en-IN"])

            # Already know is_rental, log info and continue as normal
            logger.info(f"Entities include is_rental={entities['is_rental']} for product: {product_name}")

        # ---- Existing Prompt Logic Follows ----
        prompt = f"""
    You are a textile shop assistant.
    Conversation details so far: {json.dumps(entities, ensure_ascii=False)}
    User's current intent: {intent}
    Language: {language}

    Based on the intent and info above:
    - Ask 1-2 short, relevant follow-up questions.
    - Only ask about missing or unclear details that are essential for this intent.
    - Do NOT repeat yourself or ask about already-known details.
    - Always reply ONLY with the questions (in the correct language/script).
    """
        system_content = (
            f"You are a warm, polite, and chatty textile shop owner's assistant. "
            f"Reply EVERY time in {language_name} ({language}) script, "
            f"matching the user's own language/script unless they are mixing. "
            f"Never reply in a different language."
        )
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=200,
            )
            logger.info(f"LLM Response: {response.choices[0].message.content.strip()}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error processing reply: {e}")
            return f"❌ Sorry, I couldn't process your message due to an error: {e}"

    async def analyze_message(self, text: str) -> Dict[str, Any]:
        try:
            language, lang_confidence = await detect_language(text)
            self.message_count += 1

            # Session Language Lock Logic
            N = 3  # Threshold
            if self.session_language is None:
                self.language_history.append(language)
                if len(self.language_history) == N:
                    from collections import Counter
                    lang_counts = Counter(self.language_history)
                    self.session_language = lang_counts.most_common(1)[0][0]
                # Before lock (first N turns): reply in detected language
                detected_language = language
            else:
                # After lock: always reply in locked language
                detected_language = self.session_language

            intent, new_entities, intent_confidence = await detect_textile_intent_openai(text, detected_language)
            self.merge_entities(new_entities)
            self.session_history.append({"role": "user", "content": text})

            # ---- Dynamic Intent Clarification Logic ----
            clarification_needed = (
                intent in ["other", "unknown", None] or intent_confidence < 0.7
            )
            answer = await self.generate_reply(
                detected_language, intent, self.collected_entities, text, clarification_needed=clarification_needed
            )

            self.session_history.append(
                {"role": "assistant", "content": answer})
            return {
                "input_text": text,
                "detected_language": detected_language,
                "language_confidence": lang_confidence,
                "detected_intent": intent,
                "intent_confidence": intent_confidence,
                "entities": dict(self.collected_entities),
                "answer": answer,
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
                "answer": "Sorry, something went wrong.",
                "error": str(e)
            }

    async def run_cli(self):
        print("🧵 Textile Shop Assistant - WhatsApp Style")
        print("Type 'q' or 'quit' to exit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['q', 'quit']:
                    print("Assistant: Have a great day! 👋")
                    break
                if not user_input:
                    continue
                result = await self.analyze_message(user_input)
                print(f"Assistant: {result['answer']}\n")
            except KeyboardInterrupt:
                print("\nAssistant: Session ended. Bye!")
                break
            except Exception as e:
                print(f"Assistant: ❌ Error processing your message: {e}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = TextileAnalyzer()
    asyncio.run(analyzer.run_cli())
