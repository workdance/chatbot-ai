from enum import Enum

from pydantic import BaseModel


class BrainType(str, Enum):
    DOC = "doc"
    API = "api"
    COMPOSITE = "composite"


class BrainConfig(BaseModel):
    brain_type: BrainType = BrainType.DOC
    brain_id: str = None
