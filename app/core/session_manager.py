# import redis.asyncio as aioredis
# import os
# import json

# REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# redis = aioredis.from_url(REDIS_URL, decode_responses=True)

# class SessionManager:
#     @staticmethod
#     async def get_session(user_id: str):
#         data = await redis.get(user_id)
#         return json.loads(data) if data else {}

#     @staticmethod
#     async def set_session(user_id: str, data: dict):
#         await redis.set(user_id, json.dumps(data), ex=3600)

#     @staticmethod
#     async def clear_session(user_id: str):
#         await redis.delete(user_id)
