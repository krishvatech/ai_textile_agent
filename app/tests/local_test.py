import asyncio
import logging
import os
from dotenv import load_dotenv
import httpx
import json
from datetime import datetime, timezone

# Load environment variables (if needed for auth)
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

async def simulate_incoming_whatsapp_message(url, from_number, message_body):
    # Sample payload mimicking an incoming WhatsApp message
    payload = {
        "whatsapp": {
            "messages": [
                {
                    "callback_type": "incoming_message",
                    "sid": "test_sid_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
                    "from": from_number,  # e.g., "919999999999"
                    "content": {
                        "type": "text",
                        "text": {"body": message_body}
                    }
                }
            ]
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"{datetime.now(timezone.utc)}: Simulating incoming WhatsApp message to {url}")
    # print(f"Payload: {json.dumps(payload, indent=2)}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            logging.info(f"Response from /whatsapp: {response.status_code} {response.text}")
            if response.status_code == 200:
                print("Simulation successful! Check server logs for processing details.")
                # Simulate "getting reply" by printing the server's response body (if any)
                print(f"Server Reply (if any): {response.text}")
            else:
                print("Simulation failed.")
        except Exception as e:
            logging.error(f"Error during simulation: {str(e)}")
            print("Simulation failed due to an error.")

async def main():
    url = "http://localhost:8001/whatsapp/"  # Adjust port if needed
    from_number = "911122334455"  # Replace with your test sender number
    # for Gujarati language == 914578457845
    # for Hinndi language == 911122334455
    # for English language == 
    print("Interactive WhatsApp Message Simulator. Type 'exit' to stop.")
    while True:
        message_body = input("Enter message to simulate (or 'exit' to quit): ")
        if message_body.lower() == 'exit':
            break
        await simulate_incoming_whatsapp_message(url, from_number, message_body)

# Run the simulation loop
asyncio.run(main())
