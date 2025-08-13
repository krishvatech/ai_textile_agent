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
    url = "http://localhost:8000/whatsapp/"
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