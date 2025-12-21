import argparse
import glob
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


MODEL_ID = "tencent/HunyuanOCR"


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    # bf16非対応GPUでも落ちないように自動フォールバック
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def natural_sort_key(p: str):
    parts = re.split(r"(\d+)", Path(p).name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def load_and_resize_image(path: str, long_side: int = 1280) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = long_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img


def build_prompt(processor, image_path: str, prompt_text: str) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def ocr_one(
    model,
    processor,
    image_path: str,
    prompt_text: str,
    max_new_tokens: int,
    long_side: int,
) -> str:
    img = load_and_resize_image(image_path, long_side=long_side)
    prompt = build_prompt(processor, image_path, prompt_text)

    # ★重要：truncation=False（画像トークンの不整合を避ける）
    inputs = processor(
        text=[prompt],
        images=[img],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    gen = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(gen, skip_special_tokens=True).strip()
    return text


def main():
    ap = argparse.ArgumentParser(description="Batch OCR with HunyuanOCR (folder -> TXT/JSONL)")
    ap.add_argument("--input_dir", required=True, help="画像フォルダ")
    ap.add_argument("--glob", default="*.png", help="対象パターン (例: review_page_*.png, *.jpg, *.*)")
    ap.add_argument("--prompt", default="识别图片中的文字", help="OCRプロンプト")
    ap.add_argument("--output_txt", required=True, help="まとめTXT出力パス")
    ap.add_argument("--output_jsonl", default="", help="JSONL出力パス（任意）")
    ap.add_argument("--max_new_tokens", type=int, default=1024, help="10GBなら 512〜2048目安")
    ap.add_argument("--long_side", type=int, default=1280, help="画像長辺を縮小（10GBなら1024〜1600目安）")
    args = ap.parse_args()

    dtype = pick_dtype()
    print(f"Loading {MODEL_ID} (dtype={dtype}, eager attention)...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False,
        trust_remote_code=True,
    )

    model = HunYuanVLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    pattern = str(Path(args.input_dir) / args.glob)
    paths = sorted(glob.glob(pattern), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No images matched: {pattern}")

    Path(args.output_txt).parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    fj = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None

    with open(args.output_txt, "w", encoding="utf-8") as ft:
        for i, p in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}] OCR: {p}")

            try:
                text = ocr_one(
                    model=model,
                    processor=processor,
                    image_path=p,
                    prompt_text=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    long_side=args.long_side,
                )
            except torch.cuda.OutOfMemoryError:
                # 自動リカバリ：縮小＆トークン削減で1回だけ再試行
                print("CUDA OOM -> retry with smaller settings...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                text = ocr_one(
                    model=model,
                    processor=processor,
                    image_path=p,
                    prompt_text=args.prompt,
                    max_new_tokens=max(256, args.max_new_tokens // 2),
                    long_side=max(768, args.long_side - 256),
                )
            except Exception as e:
                text = f"[ERROR] {type(e).__name__}: {e}"

            # まとめTXT（ページ区切り）
            ft.write(f"===== {p} =====\n")
            ft.write(text + "\n\n")

            # JSONL（任意）
            if fj:
                fj.write(json.dumps({"file": p, "text": text}, ensure_ascii=False) + "\n")

    if fj:
        fj.close()

    print(f"Saved TXT : {args.output_txt}")
    if args.output_jsonl:
        print(f"Saved JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
