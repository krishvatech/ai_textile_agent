import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()

EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOPHONE = os.getenv("EXOPHONE")
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")

EXOPHONE = f"+91{EXOPHONE.lstrip('0')}"

SENDER_WHATSAPP_NUMBER = f"+91{EXOPHONE.lstrip('0')}"  # e.g., "+917948516477"
TO_WHATSAPP_NUMBER = "+919726640019"   # <-- Your real test WhatsApp number here
MESSAGE_BODY = "Hello from KrishvaTech! Test via Exotel WhatsApp v2 API."

url = f"https://{EXOTEL_API_KEY}:{EXOTEL_TOKEN}@{EXOTEL_SUBDOMAIN}/v2/accounts/{EXOTEL_SID}/messages"

payload = {
    "custom_data": "Order12",
    "status_callback": "https://webhook.site",
    "whatsapp": {
        "custom_data": "Order12",
        "status_callback": "https://webhook.site",
        "messages": [
            {
                "custom_data": "Order12",
                "status_callback": "https://webhook.site",
                "from": EXOPHONE,  # Your WhatsApp-enabled sender number
                "to": "+919726640019",    # Recipient number
                "content": {
                    "recipient_type": "individual",
                    "type": "text",
                    "text": {
                        "preview_url": False,
                        "body": "MESSAGE_CONTENT"
                    }
                }
            }
            # You can add more message objects for more recipients
        ]
    }
}

print(payload)

headers = {
    "Content-Type": "application/json"
}

response = requests.post(
    url,
    data=json.dumps(payload),
    headers=headers
)

print("Status code:", response.status_code)
print("Response:", response.text)