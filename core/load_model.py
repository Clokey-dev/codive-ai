import open_clip
from ultralytics import YOLO
from rembg import new_session
import torch
import torch.nn.functional as F
from .config import DEVICE, MODEL_NAME, PRETRAINED, CLOTH_LABELS, RECORD_LABELS ,PROMPTS
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="swwho/Fashion-YOLO", filename="fashion_2.pt")

# --- 전역 변수 (Global State) ---
GLOBAL_MODEL = None
GLOBAL_PREPROCESS = None
GLOBAL_TOKENIZER = None
GLOBAL_CLOTH_EMBEDS = {}
GLOBAL_RECORD_EMBEDS = {}

GLOBAL_YOLO_MODEL = None
GLOBAL_REMBG_MODEL = None

# 함수 모드에 따른 임베딩 키 설정
EMBEDS_CONFIG_KEYS = {
    "cloth-plus": ["CATEGORIES", "COLORS", "SEASONS"],
    "record-plus": ["STYLES", "SITUATIONS"],
}



def load_clip_model():
    """
    (Lifespan에서 호출됨) 모델 로드 및 모든 임베딩을 계산하여 전역 변수에 저장
    """
    global GLOBAL_MODEL, GLOBAL_PREPROCESS, GLOBAL_TOKENIZER
    
    if GLOBAL_MODEL is not None:
        return # 이미 로드됨

    print(f"Loading CLIP model")
    
    # 1. 모델, 전처리기, 토크나이저 로드 (전역 변수에 저장)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name=MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.to(DEVICE).eval()
    
    GLOBAL_MODEL = model
    GLOBAL_PREPROCESS = preprocess
    GLOBAL_TOKENIZER = tokenizer
    
    # 2. 임베딩 계산 (제공된 load 함수의 로직)
    print("Building/Loading text embeddings...")
    with torch.no_grad():
        for name, labels in CLOTH_LABELS.items():
            emb_result = []
            for label in labels:
                embedding_prompts = [template.format(label=label) for template in PROMPTS]
                tokens = tokenizer(embedding_prompts).to(DEVICE)
                feats = model.encode_text(tokens)
                feats = F.normalize(feats, dim=-1)
                mean_feat = feats.mean(dim=0, keepdim=True)
                emb_result.append(mean_feat)
            emb_result = torch.cat(emb_result, dim=0)
            GLOBAL_CLOTH_EMBEDS[name] = (emb_result, labels) # 전역 맵에 저장

        for name, labels in RECORD_LABELS.items():
            emb_result = []
            for label in labels:
                embedding_prompts = [template.format(label=label) for template in PROMPTS]
                tokens = tokenizer(embedding_prompts).to(DEVICE)
                feats = model.encode_text(tokens)
                feats = F.normalize(feats, dim=-1)
                mean_feat = feats.mean(dim=0, keepdim=True)
                emb_result.append(mean_feat)
            emb_result = torch.cat(emb_result, dim=0)
            GLOBAL_RECORD_EMBEDS[name] = (emb_result, labels) # 전역 맵에 저장
    
    print("All resources initialized.")



def load_yolo_model(model_path: str = model_path):
    """
    YOLO 모델을 로드 함수
    """
    global GLOBAL_YOLO_MODEL

    if GLOBAL_YOLO_MODEL is not None:
        return GLOBAL_YOLO_MODEL

    print(f"Loading YOLO model from: {model_path}...")
    
    try:
        yolo_model = YOLO(model_path)
        yolo_model.to(DEVICE)
        GLOBAL_YOLO_MODEL = yolo_model
        print(f"YOLO 모델 ({model_path}) 로드 완료.")
        return GLOBAL_YOLO_MODEL
    except Exception as e:
        print(f"YOLO 모델 로드 중 오류 발생: {e}")



def load_rembg_model():
    """배경 제거 모델 로드 함수"""
    global GLOBAL_REMBG_MODEL

    if GLOBAL_REMBG_MODEL is not None:
        return
    
    print(f"Loading rembg model")

    try:
        rembg_session = new_session("u2net")
        GLOBAL_REMBG_MODEL = rembg_session
        print(f"rembg 모델 로드 완료")
        return GLOBAL_REMBG_MODEL
    except Exception as e:
        print(f"rembg 모델 로드 중 오류 발생: {e}")