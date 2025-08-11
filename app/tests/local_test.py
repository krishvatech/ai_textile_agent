import asyncio
from types import SimpleNamespace
from datetime import datetime
# Import your existing functions
from app.db.session import get_db
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai
from app.core.ai_reply import analyze_message
from sqlalchemy import text as sql_text
import logging
import os

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

EXOPHONE = os.getenv("EXOPHONE")

async def get_tenant_id_by_phone(phone_number: str, db):
    query = sql_text("SELECT id FROM tenants WHERE whatsapp_number = :phone AND is_active = true LIMIT 1")
    result = await db.execute(query, {"phone": phone_number})
    row = result.fetchone()
    return row[0] if row else None

async def whatsapp_test_local(user_text: str):
    # Database lookup for tenant ID
    async for db in get_db():
        tenant_id = await get_tenant_id_by_phone(EXOPHONE, db)
        break

    # Language detection
    current_language = None
    last_user_lang = "en-IN"  # Default
    language = await detect_language(user_text, last_user_lang)
    if isinstance(language, tuple):
        language = language[0]

    if current_language is None:
        current_language = language
        logging.info(f"Conversation language set to {current_language}")
    else:
        if current_language in ['neutral', 'en-IN'] and language in ['hi-IN', 'gu-IN']:
            current_language = language
            logging.info(f"Conversation language updated to {current_language}")

    lang = current_language
    last_user_lang = current_language

    # Intent detection
    intent_type, entities, confidence = await detect_textile_intent_openai(user_text, lang)
    logging.info(f"intent_type: {intent_type}")
    logging.info(f"entities: {entities}")
    logging.info(f"confidence: {confidence}")

    # AI Reply
    try:
        print("calling analayze message=",datetime.now())
        reply = await analyze_message(
            text=user_text,
            tenant_id=tenant_id,
            language=last_user_lang,
            intent=intent_type,
            new_entities=entities,
            intent_confidence=confidence,
            mode="chat"
        )
        print("geeting analayze message=",datetime.now())
        reply_text = reply.get("reply_text") or reply.get("answer") or "Sorry, I could not process your request right now."
    except Exception as e:
        logging.error(f"AI analyze_message failed: {e}")
        reply_text = "Sorry, our assistant is having trouble responding at the moment."

    return reply_text

async def main():
    print("📲 WhatsApp Bot Local Test — type 'q' or 'quit' to exit")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("q", "quit"):
            print("👋 Exiting local test.")
            break
        bot_reply = await whatsapp_test_local(user_input)
        print(f"Bot: {bot_reply}")

if __name__ == "__main__":
    asyncio.run(main())
