from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import WebSocket

# Initialize the FastAPI app
app = FastAPI()

# Webhook endpoint that Exotel can call
@app.get("/webhook")
async def exotel_webhook(request: Request):
    """
    This endpoint is meant to be called by Exotel as a webhook.
    It responds with a JSON containing a WebSocket URL where Exotel can stream the voice data.
    """
    # Responds with the status "ok" and the WebSocket URL for Exotel to connect to
    return JSONResponse({
        "status": "ok",
        "ws_url": "wss://krishvatech-voice-agent.onrender.com/ws"  # WebSocket URL where Exotel will connect
    })

# WebSocket endpoint for Exotel to connect and stream audio
@app.websocket("/ws")
async def voicebot_stream(websocket: WebSocket):
    """
    This endpoint will handle the WebSocket connection from Exotel, perform some actions, and close the connection.
    """
    print("Trying to connect..")  # Log the connection attempt
    await websocket.accept()  # Accept the WebSocket connection

    print("🔗 Connected to Exotel WebSocket")

    try:
        # Send a stop event to end the call immediately after playback
        await websocket.send_json({
            "event": "stop",  # Event to stop the call
            "reason": "done"  # Reason for stopping the call
        })

        await websocket.close()  # Close the WebSocket connection
        print("🔒 Call ended")
    except Exception as e:
        # Handle any errors that occur during the WebSocket connection
        print("❌ WebSocket error:", e)
        await websocket.close()  # Ensure the WebSocket connection is closed even if an error occurs
