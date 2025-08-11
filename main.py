from fastapi import FastAPI,Response
from app.api import api_router
from app.core.redis_client import init_redis, get_redis, close_redis, measure_ping_ms
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Universal AI Textile Agent")
app.include_router(api_router)

@app.on_event("startup")
async def _startup():
    # init + ping once at boot
    await init_redis()

@app.on_event("shutdown")
async def _shutdown():
    await close_redis()

@app.get("/health/redis")
async def redis_healthcheck():
    client = await get_redis()
    latency_ms = await measure_ping_ms(client)
    payload = {"status": "ok", "latency_ms": round(latency_ms, 3)}
    # Requirement: return 200 only when latency < 5ms
    if latency_ms < 5.0:
        return payload
    return Response(
        content=f'{{"status":"degraded","latency_ms":{round(latency_ms,3)}}}',
        media_type="application/json",
        status_code=503,
    )

@app.get("/")
def root():
    return {"message": "✅ API is live"}

