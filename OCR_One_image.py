import torch
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

MODEL_ID = "tencent/HunyuanOCR"

def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    # bf16非対応GPU(Turing等)だと落ちるので自動でfp16へ
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

DTYPE = pick_dtype()

def load_image(path: str, long_side: int = 1280) -> Image.Image:
    if path.startswith("http"):
        r = requests.get(path, timeout=30)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
    else:
        img = Image.open(path).convert("RGB")

    # VRAM 10GB向け：長辺を縮小（必要なら 1024 に下げてOK）
    w, h = img.size
    scale = long_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img

def main(image_path: str):
    print(f"Loading {MODEL_ID} (dtype={DTYPE}, eager attention)...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False,          # warning回避したいなら True でもOK
        trust_remote_code=True,
    )

    model = HunYuanVLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,             # torch_dtype は deprecated
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    image = load_image(image_path, long_side=1280)

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "识别图片中的文字"}  # 中国語プロンプト
        ]}
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Running OCR inference...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,   # 10GBなら 512〜2048で調整
            do_sample=False,
        )

    gen = output[0][inputs["input_ids"].shape[1]:]
    result = processor.decode(gen, skip_special_tokens=True)

    print("-" * 30)
    print("OCR Result:")
    print(result)
    print("-" * 30)

if __name__ == "__main__":
    TARGET_IMAGE = r"/mnt/c/Users/市川　裕大/Desktop/済生会川口/review_page_001.png"
    main(TARGET_IMAGE)
