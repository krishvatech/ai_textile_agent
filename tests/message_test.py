from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
import os
import requests
import logging
import httpx

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOPHONE = os.getenv("WHATSAPP_EXOPHONE")
SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")

app = FastAPI()

# To keep track of whether the message was sent already
sent_messages = {}

async def send_whatsapp_reply(to: str, body: str):
    url = f"https://{EXOTEL_API_KEY}:{EXOTEL_TOKEN}@{SUBDOMAIN}/v2/accounts/{EXOTEL_SID}/messages"

    print(f"------- sent number : {EXOPHONE}")
    payload = {
        "channel": "whatsapp",
        "whatsapp": {
            "messages": [
                {
                    "from": EXOPHONE,
                    "to": to,
                    "content": {
                        "type": "text",
                        "text": {
                            "body": body
                        }
                    }
                }
            ]
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        print(response.status_code, response.text)


@app.post("/whatsapp")
async def receive_whatsapp_message(request: Request):
    data = await request.json()
    logging.info(f"Full incoming payload: {data}")

    # ✅ Correct parsing
    incoming_msg = data.get("whatsapp", {}).get("messages", [{}])[0]
    from_number = incoming_msg.get("from", "")
    msg_type = incoming_msg.get("content", {}).get("type")
    text = ""

    if msg_type == "text":
        text = incoming_msg["content"]["text"]["body"]
    else:
        text = f"[{msg_type} message received]"

    logging.info(f"Message from {from_number}: {text}")

    reply_text = f"You said: {text}"
    await send_whatsapp_reply(to=from_number, body=reply_text)

    return {"status": "received"}

# Optional local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)