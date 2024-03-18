from fastapi import APIRouter

from app.logger import get_logger
from app.modules.brain.brain_service import BrainService

logger = get_logger(__name__)

brain_router = APIRouter()


brainService = BrainService()

@brain_router.post(
    "/brain/{brain_id}",
    tags=["Brain"]
)
async def brain_detail(brain_id: str):
    return brainService.get_brain_by_id(brain_id=brain_id)