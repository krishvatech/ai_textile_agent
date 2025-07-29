from fastapi import APIRouter, Request, BackgroundTasks, Depends
from app.db.session import get_db
from app.core.ai_agent import handle_user_message
from app.utils.exotel_utils import send_voice_response

router = APIRouter()

@router.post("/webhook")
async def voice_webhook(req: Request, background_tasks: BackgroundTasks, db = Depends(get_db)):
    body = await req.json()
    user_id = body.get("From")
    transcript = body.get("SpeechResult", "")
    tenant_id = 1  # Lookup based on incoming number/DB mapping
    shop_name = "Your Shop"
    call_sid = body.get("CallSid")
    reply_text, _, _ = await handle_user_message(user_id, transcript, tenant_id, shop_name, db, channel="voice")
    background_tasks.add_task(send_voice_response, call_sid, reply_text, "en")
    return {"status": "voice_replied"}
