import os
import shutil

from fastapi import APIRouter

from app.logger import get_logger
from app.modules.brain.brain_model import BrainReq
from app.modules.brain.brain_service import BrainService
from app.llm.vector_store import get_vectorstore_directory

logger = get_logger(__name__)

brain_router = APIRouter()

brainService = BrainService()


@brain_router.post(
    "/brain/deleteVectorStore",
    tags=["Brain"]
)
async def brain_delete_vertorstore(brainReq: BrainReq):
    brain_id = brainReq.brain_id
    folder_path = get_vectorstore_directory(brain_id)
    if (not os.path.exists(folder_path)):
        return {
            "success": True,
            "message": f"{folder_path}不存在"
        }
    try:
        shutil.rmtree(folder_path)
        logger.info(f"文件夹 {folder_path} 及其内容已删除")
    except OSError as e:
        print(f"删除文件夹 {folder_path} 失败: {e.strerror}")
    return {
        "success": True,
        "message": f"{folder_path}删除成功"
    }
