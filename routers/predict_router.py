from fastapi import APIRouter
from services.predict_services import classify_image_info, classify_image_style
from schemas.common import BaseResponse
from schemas.input_request import BaseRequest, StyleRequest
import core.load_model as load_model

router = APIRouter(prefix="/inference")


# 옷추가
@router.post("/cloth", response_model=BaseResponse)
async def classify_cloth(request_data: BaseRequest):
    """옷차림 속성 분류 엔드포인트"""
    result = await classify_image_info(
        request_data.download_url,
        request_data.upload_url,
        load_model.GLOBAL_CLOTH_EMBEDS
    )
    if result["isSuccess"]:
        return BaseResponse(
            isSuccess=True,
            message="옷 속성 분류 성공",
            result=result["result"]
        )
    return BaseResponse(
        isSuccess=False,
        message="옷 속성 분류 실패",
        error_code=result["error_code"]
    )


# 기록 추가
@router.post("/record", response_model=BaseResponse)
async def classify_record(request_data: StyleRequest):
    """상황과 스타일 분류 엔드포인트"""
    result = await classify_image_style(
        request_data.download_url,
        load_model.GLOBAL_RECORD_EMBEDS
    )
    if result["isSuccess"]:
        return BaseResponse(
            isSuccess=True,
            message="옷 스타일 분류 성공",
            result=result["result"]
        )
    return BaseResponse(
        isSuccess=False,
        message="옷 스타일 분류 실패",
        error_code=result["error_code"]
    )
