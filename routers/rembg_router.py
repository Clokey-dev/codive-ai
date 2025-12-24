from fastapi import APIRouter 
from services.preprocess_service import remove_background
from schemas.common import BaseResponse
from schemas.input_request import BaseRequest

router = APIRouter(prefix="/preprocess")

@router.put("/remove_bg", response_model=BaseResponse)
async def romve_image_background(request_data: BaseRequest):
    """presignedURL을 통해 이미지를 받아 배경이 제거된 깔끔한 이미지 업로드"""
    result = await remove_background(request_data.download_url, request_data.upload_url)
    if result["isSuccess"]:
        return BaseResponse(
            isSuccess=True,
            message="배경 제거 완료",
            result=result["result"]
        )
    return BaseResponse(
        isSuccess=False,
        message="배경 제거 실패",
        error_code=result["error_code"]
    )