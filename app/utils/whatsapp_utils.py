from fastapi import FastAPI, Request
from dotenv import load_dotenv
from fastapi import BackgroundTasks
import os
import logging
import httpx
from analyze import TextileAnalyzer
from collections import defaultdict
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

# Load environment variables
load_dotenv()

EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOPHONE = os.getenv("EXOPHONE")
SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")

# Initialize FastAPI app
app = FastAPI()

# Function to send WhatsApp reply using Exotel API
async def send_whatsapp_reply(to: str, body: str):
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
        response = await client.post(url, json=payload, headers=headers)
        logging.info(f"Sent WhatsApp reply to {to}: {body}")
        logging.info(
            f"Exotel response: {response.status_code} {response.text}")

# Endpoint to send WhatsApp message via Exotel
@app.post("/whatsapp")
async def receive_whatsapp_message(request: Request, background_tasks: BackgroundTasks):
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
        return {"status": "event-ignored"}  # Return fast for status/dlr etc

    from_number = incoming_msg.get("from", "")
    msg_type = incoming_msg.get("content", {}).get("type")
    if msg_type == "text":
        text = incoming_msg["content"]["text"]["body"]
    else:
        text = f"[{msg_type} message received]"

    logging.info(f"Message from {from_number}: {text}")
    analyzer = user_sessions[from_number]
    if text.strip().lower() in ["reset", "clear"]:
        analyzer.clear_history()
        reply_text = "👌 તમારો ચેટ રીસેટ થયો છે! પણ લખો, મજાની વાતચીત ચાલુ રાખીએ."
        background_tasks.add_task(send_whatsapp_reply, from_number, reply_text)
    else:
        async def ai_reply():
            result = await analyzer.analyze_message(text)
            reply = result.get("answer", "Sorry, unable to process your request right now.")
            await send_whatsapp_reply(from_number, reply)
        background_tasks.add_task(ai_reply)
    return {"status": "received"}