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
from app.core.chat_persistence import (
    get_or_create_customer,
    get_or_open_active_session,
    append_transcript_message,
)

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
    query = text("SELECT name FROM tenants WHERE whatsapp_number = :phone AND is_active = true LIMIT 1")
    result = await db.execute(query, {"phone": phone_number})
    row = result.fetchone()
    return row[0] if row else None

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
    if callback_type != "incoming_message" or not sid:
        return {"status": "ignored"}

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
    text = message["content"]["text"]["body"] if msg_type == "text" else f"[{msg_type} message received]"

    logging.info(f"Message from {from_number}: {text}")

    async for db in get_db():
        try:
            # 1) Resolve tenant
            tenant_id = await get_tenant_id_by_phone(EXOPHONE, db)
            if not tenant_id:
                logging.error(f"No tenant mapped to business number {EXOPHONE}. Check tenants.whatsapp_number.")
                return {"status": "no_tenant"}

            # 2) Upsert customer
            customer = await get_or_create_customer(
                db,
                tenant_id=tenant_id,
                phone=from_number,
                name=None,
                preferred_language=None,
            )

            # 3) Open (or get) active session
            chat_session = await get_or_open_active_session(db, customer_id=customer.id)

            # 4) Save inbound message
            await append_transcript_message(
                db,
                chat_session,
                role="user",
                text=text,
                msg_id=sid,
                direction="in",
                meta={"msg_type": msg_type, "raw": data},
            )

            # 5) Detect language + AI reply
            try:
                detected = await detect_language(text, "en-IN")
                current_language = detected[0] if isinstance(detected, tuple) else detected
            except Exception:
                logging.exception("Language detection failed; defaulting to en")
                current_language = "en"

            try:
                reply = await analyze_message(
                    text=text,
                    tenant_id=tenant_id,
                    channel="whatsapp",
                    language=current_language,
                    mode="chat",
                )
                reply_text = reply.get("reply_text") or reply.get("answer") or \
                            "Sorry, I could not process your request right now."
                followup_text = reply.get("followup_reply")
            except Exception:
                logging.exception("AI analyze_message failed")
                reply_text, followup_text = (
                    "Sorry, our assistant is having trouble responding at the moment. We'll get back to you soon!",
                    None,
                )

            # 6) Send replies
            await send_whatsapp_reply(to=from_number, body=reply_text)
            if followup_text:
                await send_whatsapp_reply(to=from_number, body=followup_text)

            # 7) Save outbound(s)
            await append_transcript_message(
                db, chat_session, role="assistant", text=reply_text, direction="out", meta={"reply_to": sid}
            )
            if followup_text:
                await append_transcript_message(
                    db, chat_session, role="assistant", text=followup_text, direction="out",
                    meta={"reply_to": sid, "followup": True},
                )

            # 8) Commit
            await db.commit()
        except Exception:
            logging.exception("Webhook DB flow failed; rolling back")
            await db.rollback()
            return {"status": "error"}
        finally:
            break
