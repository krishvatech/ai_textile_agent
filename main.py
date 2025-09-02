from fastapi import FastAPI,Response
from app.api import api_router
import logging
import os
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Universal AI Textile Agent")
app.include_router(api_router)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "KRISHVATECH@22"),
    same_site="lax",
    https_only=True,  # True in production
    session_cookie="txa_session",
)

# 🔑 Serve /static/* from app/static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include your API/router (your file already builds `api_router`)
app.include_router(api_router)