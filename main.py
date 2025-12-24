from fastapi import FastAPI
from contextlib import asynccontextmanager
from routers import detect_router, predict_router, rembg_router
from core.load_model import load_clip_model, load_yolo_model, load_rembg_model


# 모델 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_clip_model()
    load_yolo_model()
    load_rembg_model()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(predict_router.router)
app.include_router(rembg_router.router)
app.include_router(detect_router.router)