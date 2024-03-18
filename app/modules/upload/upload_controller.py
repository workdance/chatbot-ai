import json
import time

from fastapi import APIRouter, Request, UploadFile, HTTPException

from app.logger import get_logger
from app.util.files.file import convert_bytes, get_file_size, upload_file_storage

upload_router = APIRouter()
logger = get_logger(__name__)

@upload_router.post("/upload")
async def upload_file(
        request: Request,
        uploadFile: UploadFile,
        brain_id: str,
):
    remaining_free_space = 1000000000
    file_size = get_file_size(uploadFile)
    if file_size - remaining_free_space > 0:
        message = f"Brain will exceed maximum capacity. Maximum file allowed is : {convert_bytes(remaining_free_space)}"
        raise HTTPException(status_code=403, detail=message)
    upload_notification = None
    file_directory = str(brain_id)
    file_name = f"{int(time.time())}-{str(uploadFile.filename)}"

    try:
        file_in_storage = await upload_file_storage(uploadFile, file_directory, file_name)
        logger.info(f'File {file_in_storage} uploaded successfully')
        return {
            'success': True,
            'data': {
                'filePath': file_in_storage
            }
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }
