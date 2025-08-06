from fastapi import FastAPI, Request
from dotenv import load_dotenv
from fastapi import BackgroundTasks
import os
import logging
import httpx
from app.core.ai_reply import TextileAnalyzer
from collections import defaultdict
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Exotel config
EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOPHONE = os.getenv("EXOPHONE")
SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")

app = FastAPI()

# Session store per user
user_sessions = defaultdict(TextileAnalyzer)

async def send_whatsapp_reply(to: str, body: str) -> None:
    """Send a reply message to WhatsApp via Exotel API."""
    url = f"https://{EXOTEL_API_KEY}:{EXOTEL_TOKEN}@{SUBDOMAIN}/v2/accounts/{EXOTEL_SID}/messages"
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
        try:
            response = await client.post(url, json=payload, headers=headers)
            logging.info(f"Sent WhatsApp reply to {to}: {body}")
            logging.info(f"Exotel response: {response.status_code} {response.text}")
        except Exception as e:
            logging.error(f"Failed to send WhatsApp message: {e}")

@app.post("/whatsapp")
async def receive_whatsapp_message(request: Request, background_tasks: BackgroundTasks):
    """Webhook endpoint for receiving incoming WhatsApp messages and replying via Exotel."""
    try:
        data = await request.json()
        logging.info(f"Full incoming payload: {data}")
        # Defensive: fallback to empty if expected fields missing
        messages = data.get("whatsapp", {}).get("messages", [])
        if not messages:
            return {"status": "no-messages"}
        incoming_msg = messages[0]
        callback_type = incoming_msg.get("callback_type", "")
        if callback_type != "incoming_message":
            logging.info(f"Ignoring non-user-message event: {callback_type}")
            return {"status": "event-ignored"}

        from_number = incoming_msg.get("from", "")
        msg_type = incoming_msg.get("content", {}).get("type", "")
        text = incoming_msg["content"]["text"]["body"] if msg_type == "text" else f"[{msg_type} message received]"
        logging.info(f"Message from {from_number}: {text}")

        analyzer = user_sessions[from_number]

        async def ai_reply():
            if text.strip().lower() in ["reset", "clear"]:
                analyzer.clear_history()
                reply_text = "👌 તમારો ચેટ રીસેટ થયો છે! પણ લખો, મજાની વાતચીત ચાલુ રાખીએ."
            else:
                try:
                    result = await analyzer.analyze_message(text)
                    reply_text = result.get("answer", "Sorry, unable to process your request right now.")
                except Exception as e:
                    logging.error(f"Error during AI processing: {e}")
                    reply_text = "Sorry, unable to process your request at the moment."
            await send_whatsapp_reply(from_number, reply_text)

        # Start background reply (immediate webhook return)
        background_tasks.add_task(ai_reply)
        return {"status": "received"}

    except Exception as e:
        logging.error(f"Error in webhook: {e}")
        return {"status": "error", "detail": str(e)}