from typing import List
from PIL import Image
from io import BytesIO
import aiohttp
import asyncio
from core.s3_utils import download_image, upload_image
import core.load_model as load_model
from schemas.common import ErrorCode

async def detect_cloth(download_url:str, upload_urls:List[str]):

    async with aiohttp.ClientSession() as session:
        download = await download_image(session, download_url)

        if not download["success"]: # 이미지 다운로드 실패 시
            return {
                "success": False,
                "error_code": download["error_code"]
            }
        
        image_bytes = download["image"]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        detect_result = load_model.GLOBAL_YOLO_MODEL(image, conf=0.35)

        if not detect_result: # 인식된 옷 객체 없을 시
            return {
                "success": False,
                "error_code": ErrorCode.DETECT_FAILED
            }
        
        boxes = detect_result[0].boxes[:10]
        crop_items = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1 = max(0, min(x1, image.width - 1))
            y1 = max(0, min(y1, image.height - 1))
            x2 = max(0, min(x2, image.width))
            y2 = max(0, min(y2, image.height))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))

            buf = BytesIO()
            crop.save(buf, format="PNG", optimize=False, compress_level=1)
            crop_items.append(buf.getvalue())
        
        if not crop_items: # 크롭한 이미지 없을 시
            return {
                "success": False,
                "error_code": ErrorCode.CROP_FAILED
            }

        async def worker(idx, img):
            return await upload_image(
                session, upload_urls[idx], img, "PUT"
            )

        results = await asyncio.gather(
            *[worker(idx, item) for idx, item in enumerate(crop_items)],
            return_exceptions=True
        )
        uploaded_idxs = [idx for idx, r in enumerate(results) if r["success"]]
        if not uploaded_idxs:
            return {
                "success": False,
                "error_code": ErrorCode.DETECT_BUT_UPLOAD_FAILED
            }

        return {
            "success": True,
            "result": {
                "detected_cnt": len(crop_items),
                "uploaded_cnt": len(uploaded_idxs),
                "uploaded_idxs": uploaded_idxs
            } 
        }