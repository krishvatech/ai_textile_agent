from openai import AsyncOpenAI
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GPT_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# Generate a reply based on the user's intent and entities
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