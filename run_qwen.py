import torch
import glob
import re
import argparse
import json
from pathlib import Path
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. HunyuanOCRから引き継いだテキストクレンジング関数
# ==========================================
KANA_RE = re.compile(r"[\u3040-\u30FF]")  # ひらがな・カタカナ

def _looks_like_cn_boilerplate(line: str) -> bool:
    if not line:
        return True
    s = line.strip()

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

    if (not KANA_RE.search(s)) and len(s) <= 40:
        if re.search(r"(图片|圖片|文字|内容|內容|官方|如下|识别|辨識|结果|結果|OCR)", s):
            return True

    return False


def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]

    removed = 0
    while lines and removed < 5 and _looks_like_cn_boilerplate(lines[0]):
        lines.pop(0)
        removed += 1

    while lines and lines[0] == "":
        lines.pop(0)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines).strip()


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ==========================================
# 2. メイン処理（バッチOCR実行）
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B-Instruct-AWQ batch processing")
    parser.add_argument("--input_dir", required=True, help="画像が入っているフォルダのパス")
    parser.add_argument("--glob", default="*.png", help="対象ファイルのパターン")
    parser.add_argument("--output_txt", required=True, help="まとめTXTの出力パス")
    parser.add_argument("--output_jsonl", default=None, help="JSONLの出力パス")
    parser.add_argument(
        "--prompt", 
        default="画像内の文字をそのまま正確に書き起こしてください。表構造がある場合はマークダウン形式で出力してください。余計な解説や前置きは省き、結果のみを出力してください。",
        help="OCRに渡すプロンプト"
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成トークン数の上限")
    args = parser.parse_args()

    def natural_sort_key(p: str):
        parts = re.split(r"(\d+)", Path(p).name)
        return [int(x) if x.isdigit() else x.lower() for x in parts]

    dtype = pick_dtype()
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

    print(f"Loading {MODEL_ID} to GPU (dtype={dtype})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="cuda",
        low_cpu_mem_usage=True
    )

    pattern = str(Path(args.input_dir) / args.glob)
    image_paths = sorted(glob.glob(pattern), key=natural_sort_key)

    if not image_paths:
        print(f"エラー: {pattern} に一致する画像が見つかりませんでした。")
        return

    print(f"合計 {len(image_paths)} 枚の画像を処理します。")

    ft = open(args.output_txt, "w", encoding="utf-8")
    fj = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None

    try:
        for i, p in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] OCR: {p}")
            
            current_max_tokens = args.max_new_tokens
            current_max_pixels = 768 * 768
            success = False
            raw_output = ""

            for attempt in range(2):
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": p},
                                {"type": "text", "text": args.prompt},
                            ],
                        }
                    ]
                    
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        min_pixels=256 * 256,
                        max_pixels=current_max_pixels,
                    ).to("cuda")
                    
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=current_max_tokens)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                    
                    raw_output = processor.batch_decode(
                        generated_ids_trimmed, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
                    success = True
                    break

                except torch.cuda.OutOfMemoryError:
                    print("   [!] CUDA OOM detected! Emptying cache and retrying with reduced parameters...")
                    torch.cuda.empty_cache()
                    current_max_pixels = max(256 * 256, current_max_pixels // 2)
                    current_max_tokens = max(256, current_max_tokens // 2)
                    if attempt == 1:
                        print("   [Error] Still OOM after reduction. Skipping this image.")
                        raw_output = "ERROR: CUDA OutOfMemoryError"

            cleaned_output = clean_ocr_text(raw_output) if success else raw_output

            # TXT 保存
            ft.write(f"===== {p} =====\n")
            ft.write(cleaned_output + "\n\n")
            ft.flush()

            # JSONL 保存
            if fj:
                fj.write(json.dumps({"file": p, "text": cleaned_output}, ensure_ascii=False) + "\n")
                fj.flush()

            status_str = "完了。不要な前置きをクレンジングして保存しました。" if success else "失敗（スキップ）。"
            print(f"   -> [{i}/{len(image_paths)}] {status_str}")

    finally:
        ft.close()
        if fj:
            fj.close()

    print(f"Saved TXT : {args.output_txt}")
    if args.output_jsonl:
        print(f"Saved JSONL: {args.output_jsonl}")

if __name__ == "__main__":
    main()