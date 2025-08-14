# session_manager.py
from typing import Dict, Any

# Process-local store (no Redis)
_SESS: Dict[str, Dict[str, Any]] = {}


def _key(user_id: str, tenant_id: int | str, channel: str) -> str:
    return f"{tenant_id}:{channel.lower()}:{user_id}"


class SessionManager:
    @staticmethod
    async def get_session(
        user_id: str, *, tenant_id: int | str, channel: str
    ) -> Dict[str, Any]:
        return _SESS.get(_key(user_id, tenant_id, channel), {})

    @staticmethod
    async def set_session(
        user_id: str, data: Dict[str, Any], *, tenant_id: int | str, channel: str
    ) -> None:
        _SESS[_key(user_id, tenant_id, channel)] = dict(data)

    @staticmethod
    async def clear_session(
        user_id: str, *, tenant_id: int | str, channel: str
    ) -> None:
        _SESS.pop(_key(user_id, tenant_id, channel), None)
