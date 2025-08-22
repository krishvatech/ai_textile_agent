from fastapi import Request, APIRouter, HTTPException
import os, logging, httpx
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")

# Country code to use when numbers are given without +CC (e.g., 0XXXXXXXXX)
DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "91")

def to_e164(num: str, cc: str = DEFAULT_COUNTRY_CODE) -> str:
    if not num:
        return ""
    n = str(num).strip().replace(" ", "")
    if n.startswith("+"):
        return n
    if n.startswith("00"):
        return "+" + n[2:]
    if n.startswith("0"):
        return f"+{cc}{n.lstrip('0')}"
    # digits only, assume already includes CC, just add +
    if n.isdigit():
        return "+" + n
    return n  # last resort; don't mangle weird input

# Normalize your WhatsApp business sender number to E.164 up front
EXOPHONE = to_e164(os.getenv("WHATSAPP_EXOPHONE", "07948516477"), DEFAULT_COUNTRY_CODE)

# Static image
STATIC_IMAGE_URL = "https://images.pexels.com/photos/32713794/pexels-photo-32713794.jpeg"
STATIC_IMAGE_CAPTION = "Thanks for your message! 🙌"

async def send_whatsapp_reply(to: str, body: str):
    if not EXOPHONE or not EXOPHONE.startswith("+"):
        # Hard fail early so you see the actual configuration issue
        raise HTTPException(
            status_code=500,
            detail="WHATSAPP_EXOPHONE is not set or invalid. Set it to your onboarded sender in E.164 (e.g., +91XXXXXXXXXX)."
        )

    url = f"https://{EXOTEL_API_KEY}:{EXOTEL_TOKEN}@{SUBDOMAIN}/v2/accounts/{EXOTEL_SID}/messages"

    to_e = to_e164(to, DEFAULT_COUNTRY_CODE)
    logging.info(f"------- sender(from): {EXOPHONE} | recipient(to): {to_e}")

    text_msg = {
        "from": EXOPHONE,
        "to": to_e,
        "content": {
            "recipient_type": "individual",
            "type": "text",
            "text": {"body": body}
        }
    }

    image_msg = {
        "from": EXOPHONE,
        "to": to_e,
        "content": {
            "recipient_type": "individual",
            "type": "image",
            "image": {
                "link": STATIC_IMAGE_URL,
                "caption": STATIC_IMAGE_CAPTION
            }
        }
    }

    payload = {
        "channel": "whatsapp",
        "whatsapp": {"messages": [text_msg, image_msg]}
    }
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers=headers)
        # Better error surfacing
        try:
            j = r.json()
        except Exception:
            j = r.text
        logging.info(f"Exotel response: {r.status_code} {j}")
        r.raise_for_status()

@router.post("/")
async def receive_whatsapp_message(request: Request):
    data = await request.json()
    logging.info(f"Full incoming payload: {data}")

    # Parse incoming
    incoming_msg = data.get("whatsapp", {}).get("messages", [{}])[0]
    from_number = incoming_msg.get("from", "")
    msg_type = incoming_msg.get("content", {}).get("type")

    if msg_type == "text":
        text = incoming_msg["content"]["text"]["body"]
    else:
        text = f"[{msg_type} message received]"

    logging.info(f"Message from {from_number}: {text}")

    reply_text = f"You said: {text}"
    await send_whatsapp_reply(to=from_number, body=reply_text)

    return {"status": "received"}
