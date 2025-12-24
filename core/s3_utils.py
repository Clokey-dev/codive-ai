import aiohttp
from typing import Dict
from schemas.common import ErrorCode

async def download_image(session: aiohttp.ClientSession, url: str):
    """단일 이미지 다운로드"""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return {
                    "success": True,
                    "image": await response.read()
                }
            return {
                "success": False,
                "error_code": ErrorCode.S3_DOWNLOAD_FAILED,
                "message":"failed to download image"
            }
    except Exception as e:
        return {
            "success": False,
            # "error_code": ErrorCode.UNEXPECTED_EXCEPTION,
            "error_code": str(e)
        }


async def upload_image(session: aiohttp.ClientSession, url: str, image_bytes: bytes, method: str) -> Dict:
    """단일 이미지 업로드"""
    try:
        if method not in ("PUT", "POST"):
            return {
                "success": False,
                "error_code": ErrorCode.INVALID_REQUEST
            }
        request = session.put if method == "PUT" else session.post
        async with request(url, data=image_bytes) as response:
            if response.status in (200, 204):
                return {"success": True}
            return {
                "success": False,
                "error_code": ErrorCode.S3_UPLOAD_FAILED,
                "http_status": response.status
            }
    except Exception as e:
        return {
            "success": False,
            # "error_code": ErrorCode.UNEXPECTED_EXCEPTION,
            "error_code": str(e)
        }