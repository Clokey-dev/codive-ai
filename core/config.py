MODEL_NAME: str = "ViT-B-32"
PRETRAINED: str = "laion2b_s34b_b79k"
DEVICE: str = "cpu"
PROMPTS = ["a photo of {label}", "a person wearing {label}", "an outfit for {label}"]

CLOTH_LABELS = {
    "categories" : [
    "t-shirt", "short sleeve shirt or blouse", "long sleeve shirt or blouse", "knitwear or sweater",
    "sweatshirt", "short sleeve t-shirt", "tank top", "denim pants or jeans", "half pants", "jogger pants",
    "cotton pants", "slacks", "leggings", "mini skirt", "midi skirt", "long skirt", "onepiece dress",
    "short padding", "sheepskin jacket", "zip-up hoodie", "windbreak", "leather jacket", "denim jacket",
    "blazer", "cardigan", "anorak", "fleece", "coat", "long padding", "padding vest",
    "sneakers", "boots", "dress shoes", "sandal or slipper",
    "crossbody bag", "shoulder bag", "backpack", "tote bag", "eco bag",
    "hat", "scarf", "socks", "wristwatch", "ring or necklace or jewelry", "belt", "glasses"
    ],
    "colors" : ["white", "black", "blue", "red", "green", "beige", "gray", "brown", "navy", "yellow", "pink", "purple"],
    "seasons" : ["spring", "summer", "autumn", "winter"],
}
RECORD_LABELS = {
    "styles" : ["casual", "street", "minimal", "classic", "chic", "vintage", "formal", "sporty", "retro", "office", "outdoor"],
    "situations" : ["daily", "travel", "date", "party", "work", "exercise"]
}