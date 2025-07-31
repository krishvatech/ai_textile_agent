from fastapi import APIRouter, Request, BackgroundTasks, Depends
from app.db.session import get_db
from app.db.models import Tenant, Product
from app.core.ai_agent import handle_user_message
from app.utils.whatsapp_utils import send_whatsapp_message#, send_whatsapp_image
from sqlalchemy import text

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
        user_msg = msg.get("content", {}).get("text", {}).get("body", "")

        print('wa_to_number', wa_to_number)
        print('wa_from_number', wa_from_number)
        print('user_msg', user_msg)

        if not (wa_to_number and wa_from_number and user_msg):
            return {"error": "Missing WhatsApp data."}
    except Exception as e:
        print("Payload parsing error:", e)
        return {"error": "Invalid payload"}

    result = await db.execute(
        text("SELECT * FROM tenants WHERE whatsapp_number = :number"), {"number": wa_to_number}
    )

    print('result.....', result)
    tenant = result.fetchone()
    if not tenant:
        print('No tenant for this WhatsApp number.......')
        return {"error": "No tenant for this WhatsApp number."}

    reply_text, images, _ = await handle_user_message(
        wa_from_number, user_msg, tenant.id, tenant.name, db
    )

    print('reply_text.....', reply_text)

    for url, caption in images:
        # background_tasks.add_task(send_whatsapp_image, wa_to_number, wa_from_number, url, caption)
        pass
    background_tasks.add_task(send_whatsapp_message, wa_to_number, wa_from_number, reply_text)
    print('print.....')
    return {"status": "replied"}

