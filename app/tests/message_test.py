# from fastapi import FastAPI, HTTPException, Request
# from dotenv import load_dotenv
# import os
# import requests
# import logging
# import httpx

# load_dotenv()

# logging.basicConfig(
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     level=logging.INFO
# )

# EXOTEL_SID = os.getenv("EXOTEL_SID")
# EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
# EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
# EXOPHONE = os.getenv("WHATSAPP_EXOPHONE")
# SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")

# app = FastAPI()

# # To keep track of whether the message was sent already
# sent_messages = {}

# async def send_whatsapp_reply(to: str, body: str):
#     url = f"https://{EXOTEL_API_KEY}:{EXOTEL_TOKEN}@{SUBDOMAIN}/v2/accounts/{EXOTEL_SID}/messages"

#     print(f"------- sent number : {EXOPHONE}")
#     payload = {
#         "channel": "whatsapp",
#         "whatsapp": {
#             "messages": [
#                 {
#                     "from": EXOPHONE,
#                     "to": to,
#                     "content": {
#                         "type": "text",
#                         "text": {
#                             "body": body
#                         }
#                     }
#                 }
#             ]
#         }
#     }

#     headers = {
#         "Content-Type": "application/json"
#     }

#     async with httpx.AsyncClient() as client:
#         response = await client.post(url, json=payload, headers=headers)
#         print(response.status_code, response.text)


# @app.post("/whatsapp")
# async def receive_whatsapp_message(request: Request):
#     data = await request.json()
#     logging.info(f"Full incoming payload: {data}")

#     # ✅ Correct parsing
#     incoming_msg = data.get("whatsapp", {}).get("messages", [{}])[0]
#     from_number = incoming_msg.get("from", "")
#     msg_type = incoming_msg.get("content", {}).get("type")
#     text = ""

#     if msg_type == "text":
#         text = incoming_msg["content"]["text"]["body"]
#     else:
#         text = f"[{msg_type} message received]"

#     logging.info(f"Message from {from_number}: {text}")

#     reply_text = f"You said: {text}"
#     await send_whatsapp_reply(to=from_number, body=reply_text)

#     return {"status": "received"}

# # Optional local run
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)

import asyncio
import logging
import os
from dotenv import load_dotenv
import httpx
import json
from datetime import datetime, timezone

# Load environment variables (if needed for auth, but not directly used here)
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

async def simulate_incoming_whatsapp_message():
    # URL for the local FastAPI /whatsapp endpoint (assuming server is running on localhost:8000)
    url = "http://localhost:8001/whatsapp/"
    
    # Sample payload mimicking an incoming WhatsApp message from Exotel
    # Customize as needed (based on the structure in your conversation history)
    payload = {
        "whatsapp": {
            "messages": [
                {
                    "callback_type": "incoming_message",
                    "sid": "test_sid_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
                    "from": "919999999999",  # Replace with sender's number
                    "content": {
                        "type": "text",
                        "text": {"body": "Hello, this is a test message!"}
                    }
                }
            ]
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"{datetime.now(timezone.utc)}: Simulating incoming WhatsApp message to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        logging.info(f"Response from /whatsapp: {response.status_code} {response.text}")
        
        if response.status_code == 200:
            print("Simulation successful! Check server logs for processing details.")
        else:
            print("Simulation failed.")

# Run the simulation
asyncio.run(simulate_incoming_whatsapp_message())