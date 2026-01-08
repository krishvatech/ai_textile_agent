from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerateCatalogResult(BaseModel):
    job_id: str
    images: List[str]
    meta: Dict[str, Any] = Field(default_factory=dict)
    debug: Optional[Dict[str, Any]] = None
