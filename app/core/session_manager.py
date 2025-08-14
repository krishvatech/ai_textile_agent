# app/core/session_manager.py
from typing import Any, Dict, Union

_SESS: Dict[str, Dict[str, Any]] = {}

def _key(user_id: Union[str, int], tenant_id: Union[str, int], channel: str) -> str:
    return f"{tenant_id}:{str(channel).lower()}:{user_id}"

class SessionManager:
    @staticmethod
    async def get_session(user_id: Union[str, int], *, tenant_id: Union[str, int], channel: str) -> Dict[str, Any]:
        return _SESS.get(_key(user_id, tenant_id, channel), {})

    @staticmethod
    async def set_session(user_id: Union[str, int], data: Dict[str, Any], *, tenant_id: Union[str, int], channel: str) -> None:
        _SESS[_key(user_id, tenant_id, channel)] = dict(data)

    @staticmethod
    async def clear_session(user_id: Union[str, int], *, tenant_id: Union[str, int], channel: str) -> None:
        _SESS.pop(_key(user_id, tenant_id, channel), None)
