from fastapi import FastAPI
from app.api import api_router

app = FastAPI(title="AI Textile Agent SaaS")
app.include_router(api_router)
