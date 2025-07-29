import redis.asyncio as aioredis
import os
import json

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = aioredis.from_url(REDIS_URL, decode_responses=True)

def build_key(user_id, tenant_id, channel):
    return f"{tenant_id}:{channel}:{user_id}"

async def get_user_memory(user_id, tenant_id, channel):
    key = build_key(user_id, tenant_id, channel)
    data = await redis.get(key)
    return json.loads(data) if data else {}

async def set_user_memory(user_id, tenant_id, channel, data: dict):
    key = build_key(user_id, tenant_id, channel)
    await redis.set(key, json.dumps(data), ex=3600)
