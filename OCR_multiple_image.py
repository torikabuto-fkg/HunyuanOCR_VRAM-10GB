import argparse
import glob
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


MODEL_ID = "tencent/HunyuanOCR"

import re

KANA_RE = re.compile(r"[\u3040-\u30FF]")  # ひらがな・カタカナ

def _looks_like_cn_boilerplate(line: str) -> bool:
    if not line:
        return True
    s = line.strip()

    # 典型の前置き（簡体/繁体混在も吸う）
    fixed = [
        r"^以下是图片中的文字内容[。．\.]*$",
        r"^以下为图片中的文字内容[。．\.]*$",
        r"^以下是圖片中的文字內容[。．\.]*$",
        r"^以下為圖片中的文字內容[。．\.]*$",
        r"^图片中的文字如下[:：]?$",
        r"^图片中的文字为[:：]?$",
        r"^圖片中的文字如下[:：]?$",
        r"^圖片中的文字為[:：]?$",
        r"^图片中的文字内容[:：]?$",
        r"^圖片中的文字內容[:：]?$",
        r"^识别结果[:：]?$",
        r"^识别结果如下[:：]?$",
        r"^辨識結果[:：]?$",
        r"^辨識結果如下[:：]?$",
        r"^OCR\s*Result[:：]?$",
        r"^The text in the image.*$",
    ]
    for p in fixed:
        if re.match(p, s):
            return True

    # “前置きっぽいキーワード”を含むのに、かなが一切ない短文は前置き扱いで落とす
    # ※日本語本文にはかなが混ざることが多いので誤爆が少ない
    if (not KANA_RE.search(s)) and len(s) <= 40:
        if re.search(r"(图片|圖片|文字|内容|內容|如下|识别|辨識|结果|結果|OCR)", s):
            return True

    return False


def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]

    # 先頭の前置き行を「連続で」削る（最大5行まで）
    removed = 0
    while lines and removed < 5 and _looks_like_cn_boilerplate(lines[0]):
        lines.pop(0)
        removed += 1

    # 先頭空行も削除
    while lines and lines[0] == "":
        lines.pop(0)

    # 末尾空行も削除
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines).strip()

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
    text = processor.decode(gen, skip_special_tokens=True)
    text = clean_ocr_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="Batch OCR with HunyuanOCR (folder -> TXT/JSONL)")
    ap.add_argument("--input_dir", required=True, help="画像フォルダ")
    ap.add_argument("--glob", default="*.png", help="対象パターン (例: review_page_*.png, *.jpg, *.*)")
    ap.add_argument(
        "--prompt",
        default="画像内の文字をそのまま書き起こしてください。説明文や前置き（例：『以下は～』）は出力せず、文字の本文だけを改行を保って出力してください。日本語は日本語のまま出力してください。",
        help="OCRプロンプト"
    )
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
