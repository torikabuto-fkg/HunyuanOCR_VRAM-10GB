import torch
import glob
import re
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

# ==========================================
# 1. 定数とモデルIDの定義
# ==========================================
MODEL_ID = "tencent/HunyuanOCR"
INPUT_DIR = "./画像"
GLOB_PATTERN = "review_page_*.png"
OUTPUT_FILE = "./output_hunyuan.txt"

KANA_RE = re.compile(r"[\u3040-\u30FF]")  # ひらがな・カタカナ

# ==========================================
# 2. テキストクレンジング関数（中国語の前置きを消去）
# ==========================================
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
    if (not KANA_RE.search(s)) and len(s) <= 40:
        if re.search(r"(图片|圖片|文字|内容|內容|官方|如下|识别|辨識|结果|結果|OCR)", s):
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


def load_image(path: str, long_side: int = 1280) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = long_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ==========================================
# 3. メイン処理（バッチOCR実行）
# ==========================================
def main():
    # 自然順ソート用（001, 002... を正しく並べる）
    def natural_sort_key(p: str):
        parts = re.split(r"(\d+)", Path(p).name)
        return [int(x) if x.isdigit() else x.lower() for x in parts]

    dtype = pick_dtype()
    print(f"Loading {MODEL_ID} (dtype={dtype}, eager attention)...")

    # 1. プロセッサの読み込み（ここは今のままで完璧に通ります）
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False,
        trust_remote_code=True,
    )

    # 2. 推奨されている HunYuanVLForConditionalGeneration を直接インポートしてロード
    from transformers import HunYuanVLForConditionalGeneration

    print("Loading model to GPU via HunYuanVLForConditionalGeneration...")
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="cuda",               # グラボ（RTX A4000）上に配置
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    print("Model and Processor loaded successfully!!!")
    # 対象画像のリストアップ
    pattern = str(Path(INPUT_DIR) / GLOB_PATTERN)
    image_paths = sorted(glob.glob(pattern), key=natural_sort_key)

    if not image_paths:
        print(f"エラー: {pattern} に一致する画像が見つかりませんでした。")
        return

    print(f"合計 {len(image_paths)} 枚の画像を処理します。")

    # テキストファイルに結果を書き込んでいく
    with open(OUTPUT_FILE, "w", encoding="utf-8") as ft:
        for i, p in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] HunyuanOCR 処理中: {p}")
            
            # 5. 各画像に対するプロンプト設定（HunyuanOCR用にフォーマットを最適化）
            image = load_image(p, long_side=1280)
            
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": [
                    {"type": "image", "image": p},
                    {"type": "text", "text": "画像内の文字をそのまま正確に書き起こしてください。表構造がある場合はマークダウン形式で出力してください。余計な解説や前置きは省き、結果のみを出力してください。"}
                ]}
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 各画像のプロセッサ処理（HunyuanOCRの入力形式に対応）
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt"
            ).to("cuda", dtype=dtype)
            
            # 推論実行（出力上限を1024文字に制限して高速化）
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    do_sample=False  # 精度を安定させるために決定論的生成
                )
            
            # 結果のデコード（入力部分を切り落として出力）
            gen = generated_ids[0][inputs["input_ids"].shape[1]:]
            raw_output = processor.decode(gen, skip_special_tokens=True)
            
            # 前置きのクレンジング処理を適用
            cleaned_output = clean_ocr_text(raw_output)
            
            # 結果をリアルタイム保存
            ft.write(f"===== {p} =====\n")
            ft.write(cleaned_output + "\n\n")
            ft.flush()
            
            print(f"   -> [{i}/{len(image_paths)}] 完了。結果を保存しました。")

    print(f"すべての処理が完了しました！結果は {OUTPUT_FILE} に保存されました。")

if __name__ == "__main__":
    main()
