# session_memory.py
from typing import Dict, Any

# Process-local store (no Redis)
_STORE: Dict[str, Dict[str, Any]] = {}


def build_key(user_id: str, tenant_id: int | str, channel: str) -> str:
    """tenant + channel + user = isolated key (prevents cross-customer merge)"""
    return f"{tenant_id}:{channel.lower()}:{user_id}"


async def get_user_memory(user_id: str, tenant_id: int | str, channel: str) -> Dict[str, Any]:
    key = build_key(user_id, tenant_id, channel)
    return _STORE.get(key, {})


async def set_user_memory(
    user_id: str,
    tenant_id: int | str,
    channel: str,
    data: Dict[str, Any],
) -> None:
    key = build_key(user_id, tenant_id, channel)
    _STORE[key] = dict(data)  # shallow copy to avoid external mutations


async def clear_user_memory(user_id: str, tenant_id: int | str, channel: str) -> None:
    key = build_key(user_id, tenant_id, channel)
    _STORE.pop(key, None)
