from fastapi import FastAPI
from app.api import api_router
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Universal AI Textile Agent")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "✅ API is live"}

