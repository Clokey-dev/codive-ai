from io import BytesIO
from PIL import Image
from rembg import remove
import core.load_model as load_model
from core.s3_utils import download_image, upload_image
import aiohttp

async def remove_background_with_session(session: aiohttp.ClientSession, download_url:str, upload_url:str)->bytes:
    """
    이미지를 내려받고, 해당 이미지의 배경을 제거하여, 해당 이미지를 업로드한다.
    """
    # 이미지 다운로드
    download_response = await download_image(session, download_url)
    if download_response["success"] is False:
        return download_response
    image_bytes = download_response["image"]
    
    # 이미지 배경제거
    rembg_model = load_model.GLOBAL_REMBG_MODEL
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    removed_result = remove(image, rembg_model)

    buffer = BytesIO()
    removed_result.save(buffer, format="PNG")
    buffer.seek(0)
    removed_background_image_bytes = buffer.read()

    # 이미지 업로드
    upload_response = await upload_image(session, upload_url, removed_background_image_bytes, "PUT")
    if upload_response["success"] is False:
        return upload_response
    return upload_response


async def remove_background(download_url:str, upload_url:str):
    """단일 이미지에 대한 배경 제거 함수"""
    async with aiohttp.ClientSession() as session:
        return await remove_background_with_session(session=session, download_url=download_url, upload_url=upload_url)