from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel

class ErrorCode(str, Enum):
    UNEXPECTED_EXCEPTION = "UNEXPECTED_EXCEPTION"
    INVALID_REQUEST = "INVALID_REQUEST"
    # S3
    S3_UPLOAD_FAILED = "S3_UPLOAD_FAILED"
    S3_DOWNLOAD_FAILED = "S3_DOWNLOAD_FAILED"
    S3_PRESIGNED_URL_INVALID = "S3_PRESIGNED_URL_INVALID"
    # detect
    DETECT_FAILED = "DETECT_EMPTY"
    CROP_FAILED = "CROP_EMPTY"
    DETECT_BUT_UPLOAD_FAILED = "DETECT_BUT_UPLOAD_FAILED"


class BaseResponse(BaseModel):
    isSuccess: bool = True
    message: str = "요청이 정상적으로 처리되었습니다."
    result: Optional[Any] = None
    error_code: Optional[Any] = None
