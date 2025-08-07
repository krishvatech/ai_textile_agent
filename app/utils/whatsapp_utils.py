from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os
import logging
import httpx
from sqlalchemy import text
from app.db.session import get_db
from app.core.lang_utils import detect_language
from app.core.intent_utils import detect_textile_intent_openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

# Exotel Credentials from environment
EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOPHONE = os.getenv("EXOPHONE")
SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")

app = FastAPI()

# Simple in-memory deduplication for processed incoming message SIDs
processed_message_sids = set()

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

@app.post("/whatsapp")
async def receive_whatsapp_message(request: Request):
    """
    Handle incoming WhatsApp messages and DLR/webhook events from Exotel.
    Only process 'incoming_message' events and reply only ONCE per message SID.
    """
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
    if sid in processed_message_sids:
        logging.info(f"Duplicate incoming_message SID {sid} ignored")
        return {"status": f"duplicate_{sid}"}
    processed_message_sids.add(sid)

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
        break  # Only use one DB session

    # Run language/entity detection (awaiting)
    language = await detect_language(text)
    intent_type, entities, confidence = await detect_textile_intent_openai(text, language)
    logging.info(f"intent_type : {intent_type}")
    logging.info(f"Entities : {entities}")
    logging.info(f"confidence : {confidence}")

    reply_text = (
        f"You said: {text}\n"
        f"Language: {language}\n"
        f"Entities: {entities}\n"
        f"Your tenant ID is: {tenant_id if tenant_id else 'not found'}"
    )

    await send_whatsapp_reply(to=from_number, body=reply_text)
    return {"status": "received"}

# For optional local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
