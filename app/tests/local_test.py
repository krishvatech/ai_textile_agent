# import asyncio
# import logging
# import os
# from dotenv import load_dotenv
# import httpx
# import json
# from datetime import datetime, timezone

# # Load environment variables (if needed for auth)
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     level=logging.INFO
# )

# async def simulate_incoming_whatsapp_message(url, from_number, message_body):
#     # Sample payload mimicking an incoming WhatsApp message
#     payload = {
#         "whatsapp": {
#             "messages": [
#                 {
#                     "callback_type": "incoming_message",
#                     "sid": "test_sid_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
#                     "from": from_number,  # e.g., "919999999999"
#                     "content": {
#                         "type": "text",
#                         "text": {"body": message_body}
#                     }
#                 }
#             ]
#         }
#     }
    
#     headers = {"Content-Type": "application/json"}
    
#     print(f"{datetime.now(timezone.utc)}: Simulating incoming WhatsApp message to {url}")
#     # print(f"Payload: {json.dumps(payload, indent=2)}")
    
#     async with httpx.AsyncClient() as client:
#         try:
#             response = await client.post(url, json=payload, headers=headers)
#             logging.info(f"Response from /whatsapp: {response.status_code} {response.text}")
#             if response.status_code == 200:
#                 print("Simulation successful! Check server logs for processing details.")
#                 # Simulate "getting reply" by printing the server's response body (if any)
#                 print(f"Server Reply (if any): {response.text}")
#             else:
#                 print("Simulation failed.")
#         except Exception as e:
#             logging.error(f"Error during simulation: {str(e)}")
#             print("Simulation failed due to an error.")

# async def main():
#     url = "http://localhost:8001/whatsapp/webhook"  # Adjust port if needed

#     # Language -> test sender number mapping
#     LANG_NUMBERS = {
#         "gu": "914578457845",  # Gujarati
#         "hi": "911122334455",  # Hindi
#         "en": "917410852078",  # English
#     }

#     NAME_ALIASES = {
#         "1": "gu", "gujarati": "gu", "gu": "gu",
#         "2": "hi", "hindi": "hi", "hi": "hi",
#         "3": "en", "english": "en", "en": "en",
#     }

#     print("Interactive WhatsApp Message Simulator.")
#     print("Choose language for the test sender number:")
#     print("  1) Gujarati (gu)  -> 914578457845")
#     print("  2) Hindi (hi)     -> 911122334455")
#     print("  3) English (en)   -> 917410852078")

#     # ask until valid
#     lang_choice = None
#     while lang_choice not in ("gu", "hi", "en"):
#         raw = input("Enter 1/2/3 or gu/hi/en (default: en): ").strip().lower()
#         if not raw:
#             lang_choice = "en"
#         else:
#             lang_choice = NAME_ALIASES.get(raw)

#         if lang_choice not in ("gu", "hi", "en"):
#             print("⚠️  Invalid choice. Please try again.")

#     from_number = LANG_NUMBERS[lang_choice]
#     print(f"\n✅ Using language: {lang_choice.upper()} | From number: {from_number}")
#     print("Type 'exit' to stop.\n")

#     while True:
#         message_body = input("Enter message to simulate (or 'exit' to quit): ").strip()
#         if message_body.lower() == "exit":
#             break
#         await simulate_incoming_whatsapp_message(url, from_number, message_body)

# # Run the simulation loop
# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
import logging
import os
import time
import json
import hmac, hashlib
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────

load_dotenv()

# Point this to your local webhook route
# If your endpoint is /meta/webhook, change the default below accordingly.
WEBHOOK_URL = os.getenv("WHATSAPP_WEBHOOK_URL", "http://localhost:8001/whatsapp/webhook")

DISPLAY_PHONE_NUMBER = os.getenv("DISPLAY_PHONE_NUMBER", "919274417433")
PHONE_NUMBER_ID      = os.getenv("PHONE_NUMBER_ID", "805136529344134")

# ✅ FIXED: Enable HMAC signing by default for production-like testing
SEND_X_LOCAL_TEST = os.getenv("SEND_X_LOCAL_TEST", "1").lower() in ("1", "true", "yes")
SIGN_WITH_HMAC    = os.getenv("SIGN_WITH_HMAC", "1").lower() in ("1", "true", "yes")  # ✅ Changed default to "1"
META_APP_SECRET   = os.getenv("META_APP_SECRET", "")  # ✅ Must be set in .env file

# Validate required configuration
if SIGN_WITH_HMAC and not META_APP_SECRET:
    raise ValueError(
        "❌ META_APP_SECRET is required when SIGN_WITH_HMAC=1. "
        "Please set it in your .env file with your actual Meta App Secret."
    )

LANG_NUMBERS = {
    "gu": "914578457845",  # Gujarati
    "hi": "911122334455",  # Hindi
    "en": "917410852078",  # English
}

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _now_ts() -> str:
    return str(int(time.time()))

def _new_wamid(suffix: str = "") -> str:
    base = f"wamid.LOCAL_{int(time.time()*1000)}"
    return f"{base}_{suffix}" if suffix else base

def _base_payload(from_waid: str, name: str = "Local Tester") -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "716275061413813",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {
                        "display_phone_number": DISPLAY_PHONE_NUMBER,
                        "phone_number_id": PHONE_NUMBER_ID
                    },
                    "contacts": [{
                        "profile": {"name": name},
                        "wa_id": from_waid
                    }],
                },
                "field": "messages"
            }]
        }]
    }

def _hmac_sig(raw_bytes: bytes) -> str:
    """
    ✅ FIXED: Proper HMAC signature generation
    Matches server-side: "sha256=" + HMAC_SHA256(APP_SECRET, raw_request_body)
    """
    if not META_APP_SECRET:
        logging.warning("META_APP_SECRET is empty - signature will be invalid!")
        return ""
    
    signature = hmac.new(
        META_APP_SECRET.encode(), 
        raw_bytes, 
        hashlib.sha256
    ).hexdigest()
    
    return f"sha256={signature}"

async def _post_json(url: str, payload: dict):
    """
    ✅ IMPROVED: Better error handling and logging
    """
    # IMPORTANT: serialize once and reuse the same bytes for HMAC + request body
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    
    if SEND_X_LOCAL_TEST:
        headers["X-LOCAL-TEST"] = "1"
        
    if SIGN_WITH_HMAC:
        signature = _hmac_sig(raw)
        if signature:
            headers["X-Hub-Signature-256"] = signature
            logging.info("✅ HMAC signature added to request")
        else:
            logging.error("❌ Failed to generate HMAC signature!")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Use `content=raw` to ensure signature matches body EXACTLY
            resp = await client.post(url, content=raw, headers=headers)

        ok = 200 <= resp.status_code < 300
        body_preview = (resp.text or "")[:400]
        
        # ✅ IMPROVED: Better status reporting
        status_emoji = "✅" if ok else "❌"
        logging.info(
            "%s POST %s -> %s (local=%s, signed=%s)", 
            status_emoji, url, resp.status_code, SEND_X_LOCAL_TEST, SIGN_WITH_HMAC
        )
        
        print("─" * 80)
        print(f"Status: {resp.status_code} {status_emoji}")
        print(f"Body  : {body_preview}")
        print("Headers sent:", {k: v for k, v in headers.items() if k.lower() != "content-type"})
        
        # ✅ ADDED: Show response headers for debugging
        if not ok:
            print("Response headers:", dict(resp.headers))
        
        print("─" * 80)
        return ok, resp
        
    except Exception as e:
        logging.error("❌ Request failed: %s", e)
        print(f"❌ Request failed: {e}")
        return False, None

# ────────────────────────────────────────────────────────────────────────────────
# Flows
# ────────────────────────────────────────────────────────────────────────────────

async def simulate_normal_text(url: str, from_waid: str, text: str):
    payload = _base_payload(from_waid)
    payload["entry"][0]["changes"][0]["value"]["messages"] = [{
        "from": from_waid,
        "id": _new_wamid("USER_NORMAL"),
        "timestamp": _now_ts(),
        "text": {"body": text},
        "type": "text"
    }]
    print(f"\n📨 Sending NORMAL text from {from_waid}: {text!r}")
    success, resp = await _post_json(url, payload)
    
    # ✅ ADDED: Success/failure feedback
    if success:
        print("✅ Message sent successfully!")
    else:
        print("❌ Message failed to send!")
    
    return success

async def simulate_swipe_reply(url: str, from_waid: str, text: str, reply_to_msg_id: str):
    payload = _base_payload(from_waid)
    payload["entry"][0]["changes"][0]["value"]["messages"] = [{
        "context": {
            "from": DISPLAY_PHONE_NUMBER,  # business number that sent the original message
            "id": reply_to_msg_id
        },
        "from": from_waid,
        "id": _new_wamid("USER_SWIPE"),
        "timestamp": _now_ts(),
        "text": {"body": text},
        "type": "text"
    }]
    print(f"\n📨 Sending SWIPE-REPLY from {from_waid} to {reply_to_msg_id}: {text!r}")
    success, resp = await _post_json(url, payload)
    
    # ✅ ADDED: Success/failure feedback
    if success:
        print("✅ Reply sent successfully!")
    else:
        print("❌ Reply failed to send!")
    
    return success

# ────────────────────────────────────────────────────────────────────────────────
# Interactive CLI
# ────────────────────────────────────────────────────────────────────────────────

async def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO
    )
    
    print("🚀 WhatsApp Local Tester (normal + swipe-reply)")
    print(f"Webhook URL: {WEBHOOK_URL}")
    print(f"Dev bypass header: {SEND_X_LOCAL_TEST} | HMAC signing: {SIGN_WITH_HMAC}")
    
    # ✅ ADDED: Configuration validation feedback
    if SIGN_WITH_HMAC and META_APP_SECRET:
        print("✅ HMAC signing is properly configured")
    elif SIGN_WITH_HMAC and not META_APP_SECRET:
        print("❌ HMAC signing enabled but META_APP_SECRET is missing!")
        print("   Please set META_APP_SECRET in your .env file")
        return
    else:
        print("⚠️  HMAC signing is disabled - requests may be rejected in production")
    
    print("────────────────────────────────────────────────────────────────────────\n")

    print("Choose language / sender number:")
    print("  1) Gujarati (gu)  ->", LANG_NUMBERS["gu"])
    print("  2) Hindi    (hi)  ->", LANG_NUMBERS["hi"])
    print("  3) English  (en)  ->", LANG_NUMBERS["en"])
    idx = input("Enter 1/2/3 (default=3): ").strip() or "3"
    lang = {"1": "gu", "2": "hi", "3": "en"}.get(idx, "en")
    from_waid = LANG_NUMBERS[lang]
    print(f"✅ Using {lang.upper()} sender: {from_waid}\n")

    while True:
        print("\nWhat do you want to send?")
        print("  1) Normal text (no context)")
        print("  2) Swipe-Reply (reply to a bot msg_id)")
        print("  3) Exit")
        choice = (input("Choice: ").strip() or "1")

        if choice == "3":
            print("👋 Bye!")
            break

        if choice not in ("1", "2"):
            print("⚠️  Invalid choice, try again.")
            continue

        text = input("Enter message text: ").strip() or "I want this"

        if choice == "1":
            await simulate_normal_text(WEBHOOK_URL, from_waid, text)
        else:
            print("\nℹ️  For SWIPE-REPLY you need the *bot* message id to reply to.")
            reply_to_msg_id = input("Paste bot msg_id (starts with 'wamid.'): ").strip()
            if not reply_to_msg_id:
                print("⚠️  Missing msg_id; aborting this attempt.")
                continue
            await simulate_swipe_reply(WEBHOOK_URL, from_waid, text, reply_to_msg_id)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error("Fatal error: %s", e, exc_info=True)


