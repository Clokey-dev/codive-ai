from fastapi import APIRouter
from services.detected_service import detect_cloth
from schemas.input_request import DetectRequest
from schemas.common import BaseResponse

router = APIRouter(prefix="/detect")

    
# API 엔드포인트
@router.post("/", response_model=BaseResponse)
async def process_and_infer(request_data: DetectRequest):
    """presignedURL을 받아, 인식된 옷 객체 크롭 이미지 반환"""
    result = await detect_cloth(request_data.download_url, request_data.upload_urls)
    if result["success"]:
        return BaseResponse(
            isSuccess=True,
            message="옷 인식 성공",
            result=result["result"]
        )
    return BaseResponse(
        isSuccess=False,
        message="옷 인식 실패",
        error_code=result["error_code"]
    )