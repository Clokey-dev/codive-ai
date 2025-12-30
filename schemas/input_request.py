from pydantic import BaseModel
from typing import List

class BaseRequest(BaseModel):
    download_url: str
    upload_url: str

class BaseBatchRequest(BaseModel):
    download_urls: List[str]
    upload_urls: List[str]

class DetectRequest(BaseModel):
    download_url: str
    upload_urls: List[str]


class StyleRequest(BaseModel):
    download_url: str