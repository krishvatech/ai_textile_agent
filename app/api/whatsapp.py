from fastapi import FastAPI, Request,APIRouter
from dotenv import load_dotenv
from datetime import datetime, timedelta  
import os
import logging
import httpx
from sqlalchemy import text
from app.db.session import get_db
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai
from app.core.ai_reply import analyze_message

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logging.info(f"Conversation language remains as")

# Exotel Credentials from environment
EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOPHONE = os.getenv("EXOPHONE")
SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")

router = APIRouter()

# Simple in-memory deduplication for processed incoming message SIDs
processed_message_sids = {}
mode = "chat"

async def get_tenant_id_by_phone(phone_number: str, db):
    """
    Fetch tenant id by phone number from the database.
    """
    query = text("SELECT id FROM tenants WHERE whatsapp_number = :phone AND is_active = true LIMIT 1")
    result = await db.execute(query, {"phone": phone_number})
    row = result.fetchone()
    if row:
        return row[0]
    return None

async def get_tenant_name_by_phone(phone_number: str, db):
    """
    Fetch tenant id by phone number from the database.
    """
    query = text("SELECT name FROM tenants WHERE whatsapp_number = :phone AND is_active = true LIMIT 1")
    result = await db.execute(query, {"phone": phone_number})
    row = result.fetchone()
    if row:
        return row[0]
    return None

async def send_whatsapp_reply(to: str, body: str):
    """
    Send WhatsApp reply using Exotel API.
    """
    url = (
        f"https://{EXOTEL_API_KEY}:{EXOTEL_TOKEN}@{SUBDOMAIN}"
        f"/v2/accounts/{EXOTEL_SID}/messages"
    )
    logging.info(f"Sending WhatsApp reply to {to} via {EXOPHONE}")

    payload = {
        "channel": "whatsapp",
        "whatsapp": {
            "messages": [
                {
                    "from": EXOPHONE,
                    "to": to,
                    "content": {
                        "type": "text",
                        "text": {"body": body}
                    }
                }
            ]
        }
    }
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        logging.info(f"Exotel API Response: {response.status_code} {response.text}")

@router.post("/")
async def receive_whatsapp_message(request: Request):
    """
    Handle incoming WhatsApp messages and DLR/webhook events from Exotel.
    Only process 'incoming_message' events and reply only ONCE per message SID.
    """
    global processed_message_sids
    data = await request.json()
    logging.info(f"Full incoming payload: {data}")

    message = data.get("whatsapp", {}).get("messages", [{}])[0]
    callback_type = message.get("callback_type", "")
    sid = message.get("sid", None)

    # Only handle real incoming user messages (not delivery reports, etc.)
    if callback_type != "incoming_message":
        logging.info(f"Ignored non-user event: {callback_type}")
        return {"status": f"ignored_{callback_type}"}
    if not sid:
        logging.warning("No SID present in incoming message!")
        return {"status": "missing_sid"}

    # Deduplication: Only process each incoming_message SID once
    # if sid in processed_message_sids:
    #     logging.info(f"Duplicate incoming_message SID {sid} ignored")
    #     return {"status": f"duplicate_{sid}"}
    now = datetime.now()
    if sid in processed_message_sids and now - processed_message_sids[sid] < timedelta(minutes=5):
        logging.info(f"Duplicate incoming_message SID {sid} ignored (first seen at {processed_message_sids[sid]})")
        return {"status": f"duplicate_{sid}"}
    processed_message_sids[sid] = now  # Add or update timestamp

    # Optional: Clean up old entries to save memory
    if len(processed_message_sids) > 1000:
        processed_message_sids = {k: v for k, v in processed_message_sids.items() if now - v < timedelta(hours=1)}

    from_number = message.get("from", "")
    msg_type = message.get("content", {}).get("type")
    text = ""

    if msg_type == "text":
        text = message["content"]["text"]["body"]
    else:
        text = f"[{msg_type} message received]"

    logging.info(f"Message from {from_number}: {text}")

    # Database look-up for tenant ID
    async for db in get_db():
        tenant_id = await get_tenant_id_by_phone(EXOPHONE, db)
        tenant_name = await get_tenant_name_by_phone(EXOPHONE, db)
        break  # Only use one DB session

    # Run language/entity detection (awaiting)
    lang_code = 'en-IN'  # Default language
    current_language = None
    last_user_lang = lang_code
    language = await detect_language(text,last_user_lang)
    if isinstance(language, tuple):
        language = language[0]
    else:
        language = language
    # Fix conversation language once set, but update if neutral or English greetings only
    if current_language is None:
        current_language = language
        logging.info(f"Conversation language set to {current_language}")
    else:
        # If current_language is neutral or en-IN (greeting), update to detected_lang if meaningful
        if current_language in ['neutral', 'en-IN'] and language in ['hi-IN', 'gu-IN']:
            current_language = language
            logging.info(f"Conversation language updated to {current_language}")
        else:
            logging.info(f"Conversation language remains as {current_language}")

    lang = current_language
    last_user_lang = current_language
    intent_type, entities, confidence = await detect_textile_intent_openai(text, lang)
    logging.info(f"intent_type : {intent_type}")
    logging.info(f"Entities : {entities}")
    logging.info(f"confidence : {confidence}")


    # ---- AI-Driven Reply ----
    reply_text = ""
    followup_text = None
    try:
        reply = await analyze_message(
            text=text,
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            language=last_user_lang,
            intent=intent_type,
            new_entities=entities,
            intent_confidence=confidence,
            mode="chat"   # Because it's WhatsApp
        )
        # select appropriate response key; fallback to something plain if unexpected
        reply_text = reply.get("reply_text") or reply.get("answer") or "Sorry, I could not process your request right now."
        followup_text = reply.get("followup_reply")
    except Exception as e:
        logging.error(f"AI analyze_message failed: {e}")
        reply_text = "Sorry, our assistant is having trouble responding at the moment. We'll get back to you soon!"


    await send_whatsapp_reply(to=from_number, body=reply_text)
    if followup_text is not None:
        await send_whatsapp_reply(to=from_number,body=followup_text)
    return {"status": "received"}
