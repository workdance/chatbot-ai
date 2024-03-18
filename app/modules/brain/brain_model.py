from enum import Enum

from pydantic import BaseModel

class BrainReq(BaseModel):
    brain_id: str = None


class BrainType(str, Enum):
    BASIC = "basic"
    DOC = "doc"
    API = "api"
    COMPOSITE = "composite"


class BrainConfig(BaseModel):
    brain_type: BrainType = BrainType.DOC
    brain_id: str = None


class BrainModel(BaseModel):
    model: str = None
    brain_id: str = None
    brain_type: BrainType = BrainType.BASIC
