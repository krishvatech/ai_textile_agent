from fastapi import FastAPI,Response
from app.api import api_router
import logging
logging.basicConfig(level=logging.INFO)
from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="Universal AI Textile Agent")
# app.include_router(api_router)




def create_app() -> FastAPI:
    app = FastAPI(
        title="Textile AI Voice/Chat Agent",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/")
    def root():
        return {"message": "✅ API is live"}

    app.include_router(api_router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
