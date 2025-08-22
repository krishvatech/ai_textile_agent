# app/api/whatsapp.py
from fastapi import FastAPI, Request, Response, APIRouter
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

WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
META_APP_SECRET = os.getenv("META_APP_SECRET")
CLOUD_SENDER_NUMBER = os.getenv("CLOUD_SENDER_NUMBER")

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

#fabric by tenant_id
async def get_tenant_fabric_by_phone(phone_number: str, db):
    # 1) Find tenant id for this WhatsApp number
    tid = (await db.execute(
        text("""SELECT id FROM tenants
                WHERE whatsapp_number = :phone AND is_active = true
                LIMIT 1"""),
        {"phone": phone_number}
    )).scalar()
    if not tid:
        return []
    # 2) Helper: does table.column exist?
    async def col_exists(table: str, col: str) -> bool:
        q = text("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = :table
              AND column_name = :col
              AND table_schema = ANY (current_schemas(false))
            LIMIT 1
        """)
        return (await db.execute(q, {"table": table, "col": col})).first() is not None
    sources = []
    # product_variants.fabric (if present)
    if await col_exists("product_variants", "fabric"):
        sources.append("""
            SELECT LOWER(TRIM(pv.fabric)) AS fab
            FROM product_variants pv
            JOIN products p ON p.id = pv.product_id
            WHERE p.tenant_id = :tid
              AND pv.fabric IS NOT NULL
              AND TRIM(pv.fabric) <> ''
        """)
    if not sources:
        return []
    union_sql = " UNION ALL ".join(sources)
    final_sql = f"SELECT DISTINCT fab FROM ({union_sql}) s"
    result = await db.execute(text(final_sql), {"tid": tid})
    rows = result.fetchall() or []
    return [r[0] for r in rows]

#color by tenant_id
async def get_tenant_color_by_phone(phone_number: str, db):
    # 1) Find tenant id for this WhatsApp number
    tid = (await db.execute(
        text("""SELECT id FROM tenants
                WHERE whatsapp_number = :phone AND is_active = true
                LIMIT 1"""),
        {"phone": phone_number}
    )).scalar()
    if not tid:
        return []
    # 2) Helper: does table.column exist?
    async def col_exists(table: str, col: str) -> bool:
        q = text("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = :table
              AND column_name = :col
              AND table_schema = ANY (current_schemas(false))
            LIMIT 1
        """)
        return (await db.execute(q, {"table": table, "col": col})).first() is not None
    sources = []
    # product_variants.fabric (if present)
    if await col_exists("product_variants", "color"):
        sources.append("""
            SELECT LOWER(TRIM(pv.color)) AS fab
            FROM product_variants pv
            JOIN products p ON p.id = pv.product_id
            WHERE p.tenant_id = :tid
              AND pv.color IS NOT NULL
              AND TRIM(pv.color) <> ''
        """)
    if not sources:
        return []
    union_sql = " UNION ALL ".join(sources)
    final_sql = f"SELECT DISTINCT fab FROM ({union_sql}) s"
    result = await db.execute(text(final_sql), {"tid": tid})
    rows = result.fetchall() or []
    return [r[0] for r in rows]

#category by tenant_id
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
          AND TRIM(p.category) <> '';
    """)
    result = await db.execute(query, {"phone": phone_number})
    rows = result.fetchall() or []
    # return a simple Python list of strings
    return [r[0] for r in rows]


# occasion by tenant_id
# occasion bby tenant_id
async def get_tenant_occasion_by_phone(phone_number: str, db):
    """
    Return a lowercase, de-duplicated list of occasion names available for the tenant
    mapped to this WhatsApp number.

    Uses:
      tenants.whatsapp_number -> tenants.id
      products.tenant_id -> products.id
      product_variants.product_id -> product_variants.id
      product_variant_occasions.variant_id -> occasions.id
    Filters out empty names and (by default) inactive variants.
    """
    query = text("""
        WITH t AS (
          SELECT id
          FROM tenants
          WHERE whatsapp_number = :phone
            AND is_active = TRUE
          ORDER BY id
          LIMIT 1
        )
        SELECT DISTINCT LOWER(TRIM(o.name)) AS occ
        FROM t
        JOIN products p
          ON p.tenant_id = t.id
        JOIN product_variants pv
          ON pv.product_id = p.id
        JOIN product_variant_occasions pvo
          ON pvo.variant_id = pv.id
        JOIN occasions o
          ON o.id = pvo.occasion_id
        WHERE COALESCE(pv.is_active, TRUE) = TRUE
          AND o.name IS NOT NULL
          AND TRIM(o.name) <> '';
    """)
    result = await db.execute(query, {"phone": phone_number})
    rows = result.fetchall() or []
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


async def send_whatsapp_reply_cloud(to_waid: str, body) -> None:
    """Send a WhatsApp reply using Meta Cloud API."""
    msg = body if isinstance(body, str) else (body.get("reply_text") if isinstance(body, dict) else str(body))
    if not PHONE_NUMBER_ID or not WHATSAPP_TOKEN:
        logging.error("Cloud API envs missing: PHONE_NUMBER_ID/WHATSAPP_TOKEN")
        return

    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_waid.replace("+", "").strip(),
        "type": "text",
        "text": {"body": (msg or "")[:4096]},
    }
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, json=payload, headers=headers)
    logging.info(f"[CLOUD] Send resp: {resp.status_code} {resp.text}")


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
            tenant_categories = await get_tenant_category_by_phone(EXOPHONE, db)
            tenant_fabric = await get_tenant_fabric_by_phone(EXOPHONE,db)
            tenant_color = await get_tenant_color_by_phone(EXOPHONE,db)
            tenant_occasion = await get_tenant_occasion_by_phone(EXOPHONE, db)
            intent_type, entities, confidence = await detect_textile_intent_openai(
                    text_msg, current_language, allowed_categories=tenant_categories,allowed_fabric=tenant_fabric,allowed_color=tenant_color,allowed_occasion=tenant_occasion
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


# --- Cloud API Webhook (Meta) ----------------------------------------------
processed_meta_msg_ids = set()

@router.get("/webhook")
async def verify_cloud_webhook(request: Request):
    """Meta verification handshake: echo hub.challenge if token matches."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        try:
            return int(challenge)
        except ValueError:
            return challenge
    return Response(status_code=403)

def _valid_signature(app_secret: str, raw: bytes, header: str) -> bool:
    if not app_secret:
        return True
    if not header or not header.startswith("sha256="):
        return False
    import hmac, hashlib
    expected = "sha256=" + hmac.new(app_secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)

@router.post("/webhook")
async def receive_cloud_webhook(request: Request):
    print('Meta webhook..................')
    """Handle inbound messages from Meta Cloud API (value.messages)."""
    raw = await request.body()
    
    # Parse data first to check payload type
    try:
        data = await request.json()
    except:
        return {"status": "invalid_json"}
    
    logging.info(f"[CLOUD] Incoming payload: {data}")
    
    # Check if it's a custom payload format
    is_custom_payload = "whatsapp" in data and "messages" in data.get("whatsapp", {})
    
    # Only validate signature for official Meta webhooks
    if not is_custom_payload:
        if not _valid_signature(META_APP_SECRET, raw, request.headers.get("X-Hub-Signature-256", "")):
            logging.warning("[CLOUD] Invalid webhook signature")
            return {"status": "forbidden"}
    else:
        logging.info("[CLOUD] Custom payload detected - skipping signature validation")
    
    msgs_to_process = []
    business_number = ""
    
    # Handle standard Meta Cloud API format
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            meta = value.get("metadata", {})  # display_phone_number, phone_number_id
            msgs = value.get("messages", [])
            if msgs:
                business_number = (CLOUD_SENDER_NUMBER or meta.get("display_phone_number", "")).replace("+", "").replace(" ", "")
                msgs_to_process.extend(msgs)
    
    # Handle custom payload format (your current structure)
    if not msgs_to_process:
        whatsapp_data = data.get("whatsapp", {})
        custom_msgs = whatsapp_data.get("messages", [])
        
        if custom_msgs:
            # Use environment variable or hardcoded business number for custom format
            business_number = (CLOUD_SENDER_NUMBER or "919876543210").replace("+", "").replace(" ", "")  # Replace with your actual number
            logging.info(f"[CLOUD] Using business_number: '{business_number}' for custom payload")
            
            # Transform custom messages to standard format
            for custom_msg in custom_msgs:
                content = custom_msg.get("content", {})
                standard_msg = {
                    "id": custom_msg.get("sid"),  # Use 'sid' as message ID
                    "from": custom_msg.get("from", ""),
                    "type": content.get("type", "unknown")
                }
                
                # Handle text messages
                if standard_msg["type"] == "text":
                    standard_msg["text"] = content.get("text", {})
                
                msgs_to_process.append(standard_msg)
    
    if not msgs_to_process:
        return {"status": "no_messages"}

    # Process all messages with unified structure
    for msg in msgs_to_process:
        msg_id = msg.get("id") or msg.get("wamid")
        if msg_id in processed_meta_msg_ids:
            continue
        processed_meta_msg_ids.add(msg_id)
        if len(processed_meta_msg_ids) > 5000:
            processed_meta_msg_ids.clear()

        from_waid = (msg.get("from") or "").strip()
        mtype = msg.get("type")
        text_msg = msg.get("text", {}).get("body", "").strip() if mtype == "text" else f"[{mtype} message]"

        async for db in get_db():
            try:
                tenant_id = await get_tenant_id_by_phone(business_number, db)
                tenant_name = await get_tenant_name_by_phone(business_number, db) or "Your Shop"
                if not tenant_id:
                    logging.error(f"[CLOUD] No tenant found for business number: '{business_number}'")
                    return {"status": "no_tenant"}

                # Persist inbound
                customer = await get_or_create_customer(db, tenant_id=tenant_id, phone=from_waid)
                chat_session = await get_or_open_active_session(db, customer_id=customer.id)
                await append_transcript_message(
                    db, chat_session, role="user", text=text_msg,
                    msg_id=msg_id, direction="in", meta={"raw": data, "channel": "cloud_api"}
                )

                # Detect language (reuse your flow)
                SUPPORTED_LANGUAGES = ["gu-IN", "hi-IN", "en-IN", "en-US"]
                current_language = customer.preferred_language or "en-IN"
                if current_language not in SUPPORTED_LANGUAGES:
                    detected = await detect_language(text_msg, "en-IN")
                    current_language = detected[0] if isinstance(detected, tuple) else detected
                    await update_customer_language(db, customer.id, current_language)

                # Optional: reuse your textile intent helpers (same as Exotel path)
                tenant_categories = await get_tenant_category_by_phone(business_number, db)
                tenant_fabric    = await get_tenant_fabric_by_phone(business_number, db)
                tenant_color     = await get_tenant_color_by_phone(business_number, db)
                tenant_occasion  = await get_tenant_occasion_by_phone(business_number, db)

                try:
                    intent_type, entities, confidence = await detect_textile_intent_openai(
                        text_msg, current_language,
                        allowed_categories=tenant_categories,
                        allowed_fabric=tenant_fabric,
                        allowed_color=tenant_color,
                        allowed_occasion=tenant_occasion,
                    )
                    raw_reply = await analyze_message(
                        text=text_msg,
                        tenant_id=tenant_id,
                        tenant_name=tenant_name,
                        language=current_language,
                        intent=intent_type,
                        new_entities=entities,
                        intent_confidence=confidence,
                        mode="chat",
                        session_key=f"{tenant_id}:whatsapp:wa:{from_waid}",
                    )
                    reply_text = (
                        raw_reply.get("reply_text") if isinstance(raw_reply, dict)
                        else str(raw_reply)
                    )
                except Exception:
                    logging.exception("[CLOUD] AI pipeline failed")
                    reply_text = "Sorry, I'm having trouble. I'll get back to you shortly."
                print("reply=",reply_text)
                # Send reply via Cloud API
                await send_whatsapp_reply_cloud(to_waid=from_waid, body=reply_text)

                # Persist outbound
                await append_transcript_message(
                    db, chat_session, role="assistant", text=reply_text,
                    direction="out", meta={"reply_to": msg_id, "channel": "cloud_api"}
                )

                await db.commit()
            except Exception:
                logging.exception("[CLOUD] Webhook DB flow failed; rolling back")
                await db.rollback()
                return {"status": "error"}
            finally:
                break
    
    return {"status": "ok"}
