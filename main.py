from fastapi import FastAPI,Response,BackgroundTasks
from app.api import api_router
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Universal AI Textile Agent")
app.include_router(api_router)
from app.api.voice import load_greeting
@app.on_event("startup")
async def startup_event():
    # Import the startup function if needed
    await load_greeting()

@app.get("/")
def root():
    return {"message": "✅ API is live"}

