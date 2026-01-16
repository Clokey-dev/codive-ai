MODEL_NAME: str = "ViT-B-32"
PRETRAINED: str = "laion2b_s34b_b79k"
DEVICE: str = "cpu"
PROMPTS = ["a photo of {label}", "a person wearing {label}", "an outfit for {label}"]

CLOTH_LABELS = {
    "categories" : [
    "t-shirt", "shirt or blouse", "knitwear or sweater", "hoodie",
    "sweatshirt", "short sleeve t-shirt", "tank top", "denim pants or jeans", "half pants", "jogger pants",
    "cotton pants", "slacks", "leggings", "mini skirt", "midi skirt", "long skirt", "onepiece dress",
    "short padding", "sheepskin jacket", "zip-up hoodie", "windbreak", "leather jacket", "denim jacket",
    "blazer", "cardigan", "anorak", "fleece", "coat", "long padding", "padding vest",
    "sneakers", "boots", "dress shoes", "sandal or slipper",
    "crossbody bag", "shoulder bag", "backpack", "tote bag", "eco bag",
    "hat", "scarf", "socks", "wristwatch", "ring or necklace or jewelry", "belt", "glasses"
    ],
    "seasons" : ["spring", "summer", "autumn", "winter"],
}
RECORD_LABELS = {
    "styles" : ["casual", "street", "minimal", "classic", "chic", "vintage", "girlish", "sporty", "lovely", "office look", "highteen"],
    "situations" : ["daily", "travel", "date", "party", "work", "exercise", "festival"]
}

STYLE_EN_TO_ID = {
    "casual": 1,
    "street": 2,
    "minimal": 3,
    "classic": 4,
    "chic": 5,
    "vintage": 6,
    "girlish": 7,
    "sporty": 8,
    "lovely": 9,
    "office look": 10,
    "highteen": 11
}

STYLE_EN_TO_KO = {
    "casual": "캐주얼",
    "street": "스트릿",
    "minimal": "미니멀",
    "classic": "클래식",
    "chic": "시크",
    "vintage": "빈티지",
    "girlish": "걸리시",
    "sporty": "스포티",
    "lovely": "러블리",
    "office look": "오피스룩",
    "highteen": "하이틴"
}

SITUATION_EN_TO_ID = {
    "daily": 1,
    "travel": 2,
    "date": 3,
    "party": 4,
    "work": 5,
    "exercise": 6,
    "festival": 7
}

SITUATION_EN_TO_KO = {
    "daily": "데일리",
    "travel": "여행",
    "date": "데이트",
    "party": "파티",
    "work": "출근룩",
    "exercise": "운동",
    "festival": "축제"
}


CATEGORY_EN_TO_ID = {
    "t-shirt":8, 
    "shirt or blouse":12, "knitwear or sweater":9, "hoodie":11,
    "sweatshirt":10, "short sleeve t-shirt":13, "tank top":14, "denim pants or jeans":16, "half pants":17, "jogger pants":18,
    "cotton pants":19, "slacks":20, "leggings":21, "mini skirt":23, "midi skirt":24, "long skirt":25, "onepiece dress":26,
    "short padding":29, "sheepskin jacket":30, "zip-up hoodie":31, "windbreak":32, "leather jacket":33, "denim jacket":34,
    "blazer":35, "cardigan":36, "anorak":37, "fleece":38, "coat":39, "long padding":40, "padding vest":41,
    "sneakers":43, "boots":44, "dress shoes":45, "sandal or slipper":46,
    "crossbody bag":48, "shoulder bag":49, "backpack":50, "tote bag":51, "eco bag":52,
    "hat":54, "scarf":55, "socks":56, "wristwatch":57, "ring or necklace or jewelry":58, "belt":59, "glasses":60
}

CATEGORY_EN_TO_KO = {
    "t-shirt": "티셔츠",
    "shirt or blouse": "셔츠/블라우스",
    "knitwear or sweater": "니트/스웨터",
    "hoodie": "후드티",
    "sweatshirt": "맨투맨",
    "short sleeve t-shirt": "반팔티",
    "tank top": "나시",
    "denim pants or jeans": "청바지",
    "half pants": "반바지",
    "jogger pants": "트레이닝/조거팬츠",
    "cotton pants": "면바지",
    "slacks": "슈트팬츠/슬랙스",
    "leggings": "레깅스",
    "mini skirt": "미니스커트",
    "midi skirt": "미디스커트",
    "long skirt": "롱스커트",
    "onepiece dress": "원피스",
    "short padding": "숏패딩/헤비 아우터",
    "sheepskin jacket": "무스탕/퍼",
    "zip-up hoodie": "후드집업",
    "windbreak": "점퍼/바람막이",
    "leather jacket": "가죽자켓",
    "denim jacket": "청자켓",
    "blazer": "슈트/블레이저",
    "cardigan": "가디건",
    "anorak": "아노락",
    "fleece": "후리스/양털",
    "coat": "코트",
    "long padding": "롱패딩",
    "padding vest": "패딩조끼",
    "sneakers": "스니커즈",
    "boots": "부츠/워커",
    "dress shoes": "구두",
    "sandal or slipper": "샌들/슬리퍼",
    "crossbody bag": "메신저/크로스백",
    "shoulder bag": "숄더백",
    "backpack": "백팩",
    "tote bag": "토트백",
    "eco bag": "에코백",
    "hat": "모자",
    "scarf": "머플러",
    "socks": "양말/레그웨어",
    "wristwatch": "시계",
    "ring or necklace or jewelry": "주얼리",
    "belt": "벨트",
    "glasses": "선글라스/안경"
}

SEASON_EN_TO_ID = {
    "spring":1,
    "summer":2,
    "autumn":3,
    "winter":4
}

SEASON_EN_TO_KO = {
    "spring":"봄",
    "summer":"여름",
    "autumn":"가을",
    "winter":"겨울"
}