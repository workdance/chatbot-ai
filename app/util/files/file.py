import os

import aiofiles
from fastapi import UploadFile


def convert_bytes(bytes, precision=2):
    """Converts bytes into a human-friendly format."""
    abbreviations = ["B", "KB", "MB"]
    if bytes <= 0:
        return "0 B"
    size = bytes
    index = 0
    while size >= 1024 and index < len(abbreviations) - 1:
        size /= 1024
        index += 1
    return f"{size:.{precision}f} {abbreviations[index]}"


def get_file_size(file: UploadFile):
    # move the cursor to the end of the file
    file.file._file.seek(0, 2)  # pyright: ignore reportPrivateUsage=none
    file_size = (
        file.file._file.tell()  # pyright: ignore reportPrivateUsage=none
    )  # Getting the size of the file
    # move the cursor back to the beginning of the file
    file.file.seek(0)

    return file_size


"""
Uploads the file content to the storage with the specified file name.

Args:
    file_content: The content of the file to be uploaded.
    file_name: The name of the file.

Returns:
    None
"""


def get_file_directory(file_directory):
    return os.path.join(os.getcwd(), "temp", file_directory)


async def upload_file_storage(file, file_directory, file_name):
    target_directory = get_file_directory(file_directory)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    target_path = os.path.join(target_directory, file_name)
    try:
        # Save the file to the 'temp' directory
        # 使用 aiofiles 来异步写文件
        async with aiofiles.open(target_path, 'wb') as out_file:
            while content := await file.read(1024):  # 读取内容为块，每块大小为1024字节
                await out_file.write(content)  # 异步写入到文件
        # print(f"File uploaded successfully to {target_path}")
        return target_path
    except Exception as e:
        print(f"An error occurred while uploading the file: {str(e)}")
        return None


