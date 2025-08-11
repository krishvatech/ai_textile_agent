from __future__ import annotations
import time
from typing import Optional
from redis.asyncio import Redis, ConnectionPool
from .config import settings

_pool: Optional[ConnectionPool] = None
_client: Optional[Redis] = None

async def init_redis() -> Redis:
    global _pool, _client
    if _client:
        return _client

    dsn = settings.redis_dsn
    _pool = ConnectionPool.from_url(
        dsn,
        decode_responses=True,          # get strings instead of bytes
        max_connections=50,             # tune as needed
        health_check_interval=30,       # keep the pool fresh
    )
    _client = Redis(connection_pool=_pool)
    # warm-up ping at startup (will raise if not reachable)
    await _client.ping()
    return _client

async def get_redis() -> Redis:
    return await init_redis()

async def close_redis():
    global _pool, _client
    if _client:
        await _client.aclose()
        _client = None
    if _pool:
        await _pool.aclose()
        _pool = None

async def measure_ping_ms(client: Redis) -> float:
    t0 = time.perf_counter()
    pong = await client.ping()
    t1 = time.perf_counter()
    if pong is not True:
        # redis-py returns True on success (not "PONG") when decode_responses=True
        raise RuntimeError("Redis PING failed")
    return (t1 - t0) * 1000.0
