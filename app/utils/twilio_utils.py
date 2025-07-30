"""
Helper functions for interacting with Twilio's WhatsApp and SMS APIs.

This module centralizes Twilio-specific logic so that the rest of the
application can remain agnostic to the underlying messaging provider.
Only WhatsApp messaging is covered here; voice calls are handled via
Exotel in this project.  To enable Twilio, make sure the following
environment variables are set:

* ``TWILIO_ACCOUNT_SID`` – Your Twilio account SID
* ``TWILIO_AUTH_TOKEN`` – Your Twilio auth token
* ``TWILIO_WHATSAPP_NUMBER`` – The WhatsApp-enabled phone number in E.164 format (without the ``whatsapp:`` prefix)

Functions in this module are async-friendly.  Internally they use
``asyncio.get_event_loop().run_in_executor`` to call the synchronous
Twilio client without blocking the event loop.  You can therefore use
them with FastAPI's ``BackgroundTasks`` to fire-and-forget messages.
"""

import os
import asyncio
from typing import Optional

try:
    from twilio.rest import Client  # type: ignore
    _twilio_available = True
except Exception:
    _twilio_available = False


TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")


def _get_client() -> Client:
    if not _twilio_available:
        raise RuntimeError("The twilio package is not installed. Add 'twilio' to your requirements.")
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        raise RuntimeError("TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN is not set in the environment.")
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


async def send_whatsapp_message(to: str, body: str) -> Optional[str]:
    """Send a WhatsApp text message via Twilio.

    :param to: Destination WhatsApp number in E.164 format (digits only or with '+')
    :param body: The message body
    :returns: The SID of the created message on success, otherwise ``None``
    """
    if not TWILIO_WHATSAPP_NUMBER:
        raise RuntimeError("TWILIO_WHATSAPP_NUMBER is not configured in the environment.")
    client = _get_client()
    loop = asyncio.get_event_loop()
    try:
        message = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                body=body,
                from_=f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
                to=f"whatsapp:{to}"
            )
        )
        return message.sid
    except Exception:
        return None


async def send_whatsapp_image(to: str, media_url: str, caption: str = "") -> Optional[str]:
    """Send a WhatsApp image message via Twilio.

    :param to: Destination WhatsApp number in E.164 format
    :param media_url: A publicly reachable URL of the image to send
    :param caption: Optional message to accompany the image
    :returns: The SID of the created message on success, otherwise ``None``
    """
    if not TWILIO_WHATSAPP_NUMBER:
        raise RuntimeError("TWILIO_WHATSAPP_NUMBER is not configured in the environment.")
    client = _get_client()
    loop = asyncio.get_event_loop()
    try:
        message = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                from_=f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
                to=f"whatsapp:{to}",
                body=caption or None,
                media_url=[media_url]
            )
        )
        return message.sid
    except Exception:
        return None