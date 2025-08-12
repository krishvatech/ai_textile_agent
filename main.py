from fastapi import FastAPI,Response
from app.api import api_router
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Universal AI Textile Agent")
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "✅ API is live"}