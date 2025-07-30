from fastapi import APIRouter, Request, BackgroundTasks, Depends
from app.db.session import get_db
from app.db.models import Tenant, Product
from app.core.ai_agent import handle_user_message
# from app.utils.whatsapp_utils import send_whatsapp_message, send_whatsapp_image

router = APIRouter()

@router.post("/webhook")
async def whatsapp_webhook(req: Request, background_tasks: BackgroundTasks, db = Depends(get_db)):
    body = await req.json()
    entry = body["entry"][0]["changes"][0]["value"]
    message = entry.get("messages", [])[0]
    wa_to_number = entry["metadata"]["display_phone_number"]
    wa_from_number = message["from"]
    user_msg = message.get("text", {}).get("body", "")
    phone_number_id = entry["metadata"]["phone_number_id"]

    result = await db.execute(
        "SELECT * FROM tenants WHERE whatsapp_number = :number", {"number": wa_to_number}
    )
    tenant = result.fetchone()
    if not tenant:
        return {"error": "No tenant for this WhatsApp number."}

    reply_text, images, _ = await handle_user_message(wa_from_number, user_msg, tenant.id, tenant.name, db)
    # for url, caption in images:
    #     background_tasks.add_task(send_whatsapp_image, phone_number_id, wa_from_number, url, caption)
    # background_tasks.add_task(send_whatsapp_message, phone_number_id, wa_from_number, reply_text)
    return {"status": "replied"}
