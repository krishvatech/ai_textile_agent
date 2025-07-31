from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
import requests
import logging
load_dotenv()
# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO    # Change to DEBUG for even more detail
)
# Load environment variables
EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOPHONE = os.getenv("EXOPHONE")
SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")
TO_NUMBER = "918799559020"
BODY = "Hello! This is a test message from the AI Textile Agent."

# Initialize FastAPI app
app = FastAPI()

# Endpoint to send WhatsApp message via Exotel
@app.post("/whatsapp")
async def send_whatsapp_message():
    url = f"https://{SUBDOMAIN}/v2/accounts/{EXOTEL_SID}/messages"
    headers = { "Content-Type": "application/json" }
    req_json = {
        "whatsapp": {
            "messages": [
                {
                    "from": EXOPHONE,
                    "to": TO_NUMBER,
                    "content": {
                        "recipient_type": "individual",
                        "type": "text",
                        "text": {
                            "preview_url": False,
                            "body": BODY
                        }
                    }
                }
            ]
        }
    }
    # Log the request details
    logging.info(f"Sending message to: {TO_NUMBER} | Body: '{BODY}'")
    logging.debug(f"POST URL: {url}")
    logging.debug(f"Request JSON: {req_json}")
    # Send the request to Exotel
    response = requests.post(url, headers=headers, json=req_json, auth=(EXOTEL_API_KEY, EXOTEL_TOKEN))
    # Log the response details
    logging.info(f"Exotel Response Status: {response.status_code}")
    logging.info(f"Exotel Response Body: {response.text}")
    if response.status_code in [200, 201, 202]:
        return {"status": "success", "response": response.json()}
    else:
        logging.error("Error sending message to Exotel")
        raise HTTPException(status_code=response.status_code, detail=response.text)
