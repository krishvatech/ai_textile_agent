from fastapi import FastAPI
from app.api import api_router

app = FastAPI(title="Universal AI Textile Agent")
app.include_router(api_router)
