from fastapi import APIRouter, Request, BackgroundTasks, Depends
from app.db.session import get_db
from app.db.models import Tenant, Product
from app.core.ai_agent import handle_user_message
from app.utils.whatsapp_utils import send_whatsapp_reply#, send_whatsapp_image
from sqlalchemy import text
from app.core.chat_persistence import (
    get_or_create_customer,
    get_or_open_active_session,
    append_transcript_message,
)

router = APIRouter()

@router.post("/webhook")
async def whatsapp_webhook(req: Request, background_tasks: BackgroundTasks, db = Depends(get_db)):
    body = await req.json()
    print("Exotel WhatsApp webhook received:", body)

    # Unpack message
    try:
        msg = body["whatsapp"]["messages"][0]
        wa_to_number = msg["to"]
        wa_from_number = msg["from"]
        # user_msg = msg.get("content", {}).get("text", {}).get("body", "")
        content = msg.get("content") or {}
        user_msg = content.get("text", {}).get("body") if msg_type == "text" else f"[{msg_type} message]"
        msg_type = content.get("type")
        sid = msg.get("sid")
        
        print('wa_to_number', wa_to_number)
        print('wa_from_number', wa_from_number)
        print('user_msg', user_msg)

        if not (wa_to_number and wa_from_number and user_msg):
            return {"error": "Missing WhatsApp data."}
    except Exception as e:
        print("Payload parsing error:", e)
        return {"error": "Invalid payload"}

    # result = await db.execute(
    #     text("SELECT * FROM tenants WHERE whatsapp_number = :number"), {"number": wa_to_number}
    # )

    result = await db.execute(
        text("SELECT * FROM tenants WHERE whatsapp_number = :number AND is_active = true LIMIT 1"),
        {"number": wa_to_number}
    )

    print('result.....', result)
    tenant = result.fetchone()
    if not tenant:
        print('No tenant for this WhatsApp number.......')
        return {"status": "no_tenant_for_whatsapp_number"}

    print('reply_text.....', reply_text)

    # ---- Persist inbound (user) message
    customer = await get_or_create_customer(db, tenant.id, phone=wa_from_number, whatsapp_id=wa_from_number)
    chat_session = await get_or_open_active_session(db, customer.id)
    await append_transcript_message(
        db, chat_session,
        role="user", text=user_msg, msg_id=sid,
        meta={"channel":"whatsapp","from":wa_from_number,"to":wa_to_number,"tenant_id":tenant.id,"vendor":"exotel"}
    )
    await db.commit()

    # ---- Generate reply from your AI
    reply_text, images, _ = await handle_user_message(
        wa_from_number, user_msg, tenant.id, tenant.name, db
    )

    background_tasks.add_task(send_whatsapp_reply, wa_to_number, wa_from_number, reply_text)
    
    await append_transcript_message(
        db, chat_session,
        role="assistant", text=reply_text,
        meta={"channel":"whatsapp","from":wa_to_number,"to":wa_from_number,"tenant_id":tenant.id,"vendor":"exotel"}
    )
    await db.commit()

    return {"status": "replied"}

