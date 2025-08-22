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

    # Language -> test sender number mapping
    LANG_NUMBERS = {
        "gu": "914578457845",  # Gujarati
        "hi": "911122334455",  # Hindi
        "en": "917410852078",  # English
    }

    NAME_ALIASES = {
        "1": "gu", "gujarati": "gu", "gu": "gu",
        "2": "hi", "hindi": "hi", "hi": "hi",
        "3": "en", "english": "en", "en": "en",
    }

    print("Interactive WhatsApp Message Simulator.")
    print("Choose language for the test sender number:")
    print("  1) Gujarati (gu)  -> 914578457845")
    print("  2) Hindi (hi)     -> 911122334455")
    print("  3) English (en)   -> 917410852078")

    # ask until valid
    lang_choice = None
    while lang_choice not in ("gu", "hi", "en"):
        raw = input("Enter 1/2/3 or gu/hi/en (default: en): ").strip().lower()
        if not raw:
            lang_choice = "en"
        else:
            lang_choice = NAME_ALIASES.get(raw)

        if lang_choice not in ("gu", "hi", "en"):
            print("⚠️  Invalid choice. Please try again.")

    from_number = LANG_NUMBERS[lang_choice]
    print(f"\n✅ Using language: {lang_choice.upper()} | From number: {from_number}")
    print("Type 'exit' to stop.\n")

    while True:
        message_body = input("Enter message to simulate (or 'exit' to quit): ").strip()
        if message_body.lower() == "exit":
            break
        await simulate_incoming_whatsapp_message(url, from_number, message_body)

# Run the simulation loop
if __name__ == "__main__":
    asyncio.run(main())