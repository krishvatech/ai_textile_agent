# app/core/session_memory.py
from typing import Any, Dict, Union

_STORE: Dict[str, Dict[str, Any]] = {}

def build_key(user_id: Union[str, int], tenant_id: Union[str, int], channel: str) -> str:
    return f"{tenant_id}:{str(channel).lower()}:{user_id}"

async def get_user_memory(user_id: Union[str, int], tenant_id: Union[str, int], channel: str) -> Dict[str, Any]:
    return _STORE.get(build_key(user_id, tenant_id, channel), {})

async def set_user_memory(user_id: Union[str, int], tenant_id: Union[str, int], channel: str, data: Dict[str, Any]) -> None:
    _STORE[build_key(user_id, tenant_id, channel)] = dict(data)

async def clear_user_memory(user_id: Union[str, int], tenant_id: Union[str, int], channel: str) -> None:
    _STORE.pop(build_key(user_id, tenant_id, channel), None)
