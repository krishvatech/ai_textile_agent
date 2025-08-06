from fastapi import FastAPI,Response
from app.api import api_router
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Universal AI Textile Agent")
app.include_router(api_router)

@app.post("/exotel/start_voice/957897")
async def exotel_start_voice():
    # Your custom message here
    message = "Hello, I am from KrishvaTech. How can I help you today?"

    xml_response = f"""
    <Response>
        <Say voice="alice" language="en-IN">Hello, I am from KrishvaTech. How can I help you today?</Say>
        <!-- <Hangup/> -->
    </Response>
    """
    return Response(content=xml_response.strip(), media_type="application/xml")

@app.get("/")
def root():
    return {"message": "✅ API is live"}

