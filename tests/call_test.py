import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/stream"
    async with websockets.connect(uri) as websocket:
        # Send connected event with phone number
        await websocket.send(json.dumps({
            "event": "connected",
            "phone_number": "1234567890"
        }))
        # Send start event with stream_sid
        await websocket.send(json.dumps({
            "event": "start",
            "stream_sid": "test123"
        }))
        # Listen for responses
        while True:
            response = await websocket.recv()
            print("Received:", response)

asyncio.run(test_ws())
