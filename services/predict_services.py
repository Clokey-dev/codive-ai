import torch
import torch.nn.functional as F
from PIL import Image
import aiohttp
import io
from core.s3_utils import download_image, upload_image
import core.load_model as load_model
from rembg import remove


# 정보 분류 함수 구현
@torch.no_grad()
async def classify_info_with_session(session: aiohttp.ClientSession, download_url: str, upload_url: str, embed_dict: dict, top_k:int=1):
    """
    이미지를 내려받고, 이미지의 정보를 추출하여, 그 결과를 반환한다.
    """
    # 이미지 내려받기
    download = await download_image(session, download_url)
    if not download["success"]:
        return {
            "isSuccess": False,
            "error_code": download["error_code"]
        }
    
    image_bytes = download["image"]

    # 이미지 배경 제거
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    removed = remove(image, load_model.GLOBAL_REMBG_MODEL)

    buffer = io.BytesIO()
    removed.save(buffer, format="PNG")
    buffer.seek(0)

    # 배경제거 이미지 업로드
    upload = await upload_image(session, upload_url, buffer.read(), "PUT")
    if not upload["success"]:
        return {
            "isSuccess": False,
            "error_code": upload["error_code"]
        }

    # 이미지 정보 추출
    img_tensor = load_model.GLOBAL_PREPROCESS(image).unsqueeze(0).to(load_model.DEVICE)
    img_feats = F.normalize(load_model.GLOBAL_MODEL.encode_image(img_tensor), dim=-1)
    logit_scale = load_model.GLOBAL_MODEL.logit_scale.exp()

    labels = {}
    for name, (embeds, texts) in embed_dict.items():
        logits = logit_scale * img_feats @ embeds.T
        _, idx = logits.topk(top_k, dim=-1)
        labels[name] = [texts[i] for i in idx.squeeze(0).tolist()]

    return {
        "isSuccess":True,
        "result": labels
    }


async def classify_image_info(download_url:str, upload_url: str, embed_type:dict, top_k:int=1):
    """단일 이미지에 대한 카테고리, 색상, 계절 정보 추출"""
    async with aiohttp.ClientSession() as session:
        return await classify_info_with_session(session, download_url, upload_url, embed_type, top_k)


# 상황 분류 함수 구현
@torch.no_grad()
async def classify_style_with_session(session: aiohttp.ClientSession, download_url: str, embed_dict: dict, top_k:int=3):
    """
    이미지를 내려받고, 가장 유사한 스타일과 상황을 추출하여, 그 결과를 반환한다.
    """
    # 이미지 내려받기
    download = await download_image(session, download_url)
    if not download["success"]:
        return {
            "isSuccess": False,
            "error_code": download["error_code"]
        }
    image_bytes = download["image"]

    # 이미지 정보 추출
    img_tensor = load_model.GLOBAL_PREPROCESS(Image.open(io.BytesIO(image_bytes)).convert("RGB")).unsqueeze(0).to(load_model.DEVICE)
    img_feats = F.normalize(load_model.GLOBAL_MODEL.encode_image(img_tensor), dim=-1)
    logit_scale = load_model.GLOBAL_MODEL.logit_scale.exp()

    labels = {}
    for name, (embeds, texts) in embed_dict.items():
        logits = logit_scale * img_feats @ embeds.T
        _, idx = logits.topk(top_k, dim=-1)
        labels[name] = [texts[i] for i in idx.squeeze(0).tolist()]

    return {
        "isSuccess":True,
        "result": labels
    }


async def classify_image_style(download_url:str, embed_type:dict, top_k:int=3):
    """단일 이미지에 대한 카테고리, 색상, 계절 정보 추출"""
    async with aiohttp.ClientSession() as session:
        return await classify_style_with_session(session, download_url, embed_type, top_k)