# app/api/whatsapp.py
from fastapi import FastAPI, Request, APIRouter
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
logging.info("Conversation language remains as")

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


async def update_customer_language(db, customer_id: int, language: str):
    """
    Update the preferred_language for a customer in the database.
    """
    query = text("UPDATE customers SET preferred_language = :language WHERE id = :customer_id")
    await db.execute(query, {"language": language, "customer_id": customer_id})
    # No commit here; it's handled in the main flow


async def get_tenant_name_by_phone(phone_number: str, db):
    query = text("SELECT name FROM tenants WHERE whatsapp_number = :phone AND is_active = true LIMIT 1")
    result = await db.execute(query, {"phone": phone_number})
    row = result.fetchone()
    return row[0] if row else None


async def get_tenant_category_by_phone(phone_number: str, db):
    """
    Return a lowercase, de-duplicated list of category names available for the tenant
    mapped to this WhatsApp number. Pulls from products.category and products.type.
    """
    query = text("""
        WITH t AS (
          SELECT id
          FROM tenants
          WHERE whatsapp_number = :phone AND is_active = true
          LIMIT 1
        )
        -- pull from products.category
        SELECT DISTINCT LOWER(TRIM(p.category)) AS cat
        FROM products p, t
        WHERE p.tenant_id = t.id
          AND p.category IS NOT NULL
          AND TRIM(p.category) <> ''
        UNION
        -- also accept 'type' as a category source (some catalogs use it)
        SELECT DISTINCT LOWER(TRIM(p.type)) AS cat
        FROM products p, t
        WHERE p.tenant_id = t.id
          AND p.type IS NOT NULL
          AND TRIM(p.type) <> '';
    """)
    result = await db.execute(query, {"phone": phone_number})
    rows = result.fetchall() or []
    # return a simple Python list of strings
    return [r[0] for r in rows]


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
        # logging.info(f"Exotel API Response: {response.status_code} {response.text}")


# ---------- NEW HELPERS: direct product pick ----------
async def get_tenant_products_by_phone(phone_number: str, db):
    """
    Return list[(product_id, product_name)] for the tenant mapped to business WhatsApp number.
    Only non-empty names are returned.
    """
    q = text("""
        WITH t AS (
          SELECT id FROM tenants WHERE whatsapp_number = :phone AND is_active = true LIMIT 1
        )
        SELECT p.id, p.name
        FROM products p, t
        WHERE p.tenant_id = t.id
          AND p.name IS NOT NULL
          AND TRIM(p.name) <> ''
    """)
    rows = (await db.execute(q, {"phone": phone_number})).fetchall() or []
    return [(r[0], r[1]) for r in rows]


# async def try_resolve_direct_pick(db, phone_number: str, message: str, threshold: int = 90):
#     """
#     If the user message looks like a direct selection of a product title,
#     resolve it to product_id using fuzzy match against tenant's product names.
#     Returns:
#       dict | None, e.g. {"intent_type": "direct_product_pick",
#                           "entities": {"product_id": 123, "product_name": "Exact Name"}}
#     """
#     prods = await get_tenant_products_by_phone(phone_number, db)
#     if not prods:
#         return None

#     names = [n for _, n in prods if n]
#     # Use full message for fuzzy match — downstream flow will confirm missing details
#     best = process.extractOne(message, names, scorer=fuzz.WRatio)
#     if best and best[1] >= threshold:
#         title = best[0]
#         pid = next(i for (i, n) in prods if n == title)
#         return {"intent_type": "direct_product_pick", "entities": {"product_id": pid, "product_name": title}}
#     return None
# -----------------------------------------------------


@router.post("/")
async def receive_whatsapp_message(request: Request):
    """
    Handle incoming WhatsApp messages and DLR/webhook events from Exotel.
    Only process 'incoming_message' events and reply only ONCE per message SID.
    """
    global processed_message_sids
    SUPPORTED_LANGUAGES = ["gu-IN", "hi-IN", "en-IN", "en-US"]
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
    text_msg = message["content"]["text"]["body"] if msg_type == "text" else f"[{msg_type} message received]"

    logging.info(f"Message from {from_number}: {text_msg}")

    # 🔑 NEW: build a per-customer session key (tenant + channel + sender-id)
    channel = "whatsapp"                          # fixed channel tag
    external_user_id = f"wa:{from_number}"        # canonical sender id (NOT EXOPHONE)
    # tenant_id we will resolve below (based on business number)

    async for db in get_db():
        try:
            # 1) Resolve tenant (mapped to your business EXOPHONE)
            tenant_id = await get_tenant_id_by_phone(EXOPHONE, db)
            if not tenant_id:
                logging.error(f"No tenant mapped to business number {EXOPHONE}. Check tenants.whatsapp_number.")
                return {"status": "no_tenant"}

            # ✅ build final session_key now (after we know tenant_id)
            session_key = f"{tenant_id}:{channel}:{external_user_id}"
            logging.info(f"[SESSION_KEY] {session_key}")   # debug: remove in prod

            # 2) Upsert customer (per-sender)
            customer = await get_or_create_customer(
                db,
                tenant_id=tenant_id,
                phone=from_number,
                name=None,
                preferred_language=None,
            )

            # 3) Open (or get) active session (DB-level isolation per customer)
            chat_session = await get_or_open_active_session(db, customer_id=customer.id)

            # 4) Save inbound message
            await append_transcript_message(
                db,
                chat_session,
                role="user",
                text=text_msg,
                msg_id=sid,
                direction="in",
                meta={
                    "msg_type": msg_type,
                    "raw": data,
                    "session_key": session_key,      # optional: for audit
                },
            )

            # 5) Detect language + AI reply
            current_language = customer.preferred_language or "en-IN"
            if current_language not in SUPPORTED_LANGUAGES:
                logging.info("Detecting language...")
                detected = await detect_language(text_msg, "en-IN")
                current_language = detected[0] if isinstance(detected, tuple) else detected
                await update_customer_language(db, customer.id, current_language)

            logging.info(f"Using language: {current_language}")

            tenant_name = await get_tenant_name_by_phone(EXOPHONE, db) or "Your Shop"

            # 🔥 NEW: Try direct product pick BEFORE intent LLM call
            # direct_pick = await try_resolve_direct_pick(db, EXOPHONE, text_msg)
            # if direct_pick:
            #     intent_type = "direct_product_pick"
            #     entities = direct_pick["entities"]
            #     confidence = 0.99
            # else:
            tenant_categories = await get_tenant_category_by_phone(EXOPHONE, db)
            intent_type, entities, confidence = await detect_textile_intent_openai(
                    text_msg, current_language, allowed_categories=tenant_categories
                )

            try:
                # 🔑 Pass the per-customer session_key to analyze_message
                raw_reply = await analyze_message(
                    text=text_msg,
                    tenant_id=tenant_id,
                    tenant_name=tenant_name,
                    language=current_language,
                    intent=intent_type,
                    new_entities=entities,
                    intent_confidence=confidence,
                    mode="chat",             # important for WhatsApp
                    session_key=session_key  # ✅ isolates memory/entities
                )
                print("Message................................!")
                print("message=", text_msg)
                print('reply..................................!')
                reply = raw_reply if isinstance(raw_reply, dict) else {"reply_text": str(raw_reply)}

                reply_text = reply.get("reply_text") or reply.get("answer") \
                             or "Sorry, I could not process your request right now."
                followup_text = reply.get("followup_reply")
                media_urls = reply.get("media") or []
            except Exception:
                logging.exception("AI analyze_message failed")
                reply_text, followup_text = (
                    "Sorry, our assistant is having trouble responding at the moment. We'll get back to you soon!",
                    None,
                )
            print("reply=", reply_text)
            print('reply..................................!')
            print("Followup=", followup_text)

            # 6) Send replies
            await send_whatsapp_reply(to=from_number, body=reply_text)
            if followup_text:
                await send_whatsapp_reply(to=from_number, body=followup_text)

            # 7) Save outbound(s)
            await append_transcript_message(
                db, chat_session, role="assistant", text=reply_text, direction="out",
                meta={"reply_to": sid, "session_key": session_key}  # optional
            )
            if followup_text:
                await append_transcript_message(
                    db, chat_session, role="assistant", text=followup_text, direction="out",
                    meta={"reply_to": sid, "followup": True, "session_key": session_key},
                )

            # 8) Commit
            await db.commit()
        except Exception:
            logging.exception("Webhook DB flow failed; rolling back")
            await db.rollback()
            return {"status": "error"}
        finally:
            break
