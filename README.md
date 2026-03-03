# HunyuanOCR_VRAM-10GB

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-RTX%203080%2010GB-green?logo=nvidia)
![Model](https://img.shields.io/badge/Model-HunyuanOCR-orange)

**RTX 3080 (VRAM 10GB) 環境で [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) を動作させるための軽量化・最適化リポジトリ。**

Tencent の HunyuanOCR は高精度な VLM ベースの OCR モデルですが、公式のままでは VRAM 10GB では OOM になります。  
本リポジトリでは **画像縮小・dtype 自動選択・OOM 自動リカバリ** などの最適化により、RTX 3080 でも安定動作を実現しています。

---

## ✨ 特徴

| 機能 | 説明 |
|------|------|
| **VRAM 10GB 対応** | 画像長辺の自動縮小 + bf16/fp16 自動選択で 10GB GPU に収まるよう最適化 |
| **OOM 自動リカバリ** | CUDA OOM 発生時に画像サイズ・トークン数を自動縮小して 1 回リトライ |
| **中国語前置き除去** | HunyuanOCR が出力しがちな中国語の前置き文（「以下是图片中的文字内容」等）を自動フィルタ |
| **日本語最適化プロンプト** | 日本語テキストをそのまま書き起こすカスタムプロンプトを使用 |
| **自然順ソート** | `review_page_1.png` → `review_page_2.png` → ... → `review_page_10.png` の正しい順序で処理 |
| **複数出力形式** | TXT（ページ区切り付き）+ JSONL（プログラム連携用） |

---

## 📁 ファイル構成

```
HunyuanOCR_VRAM-10GB/
├── OCR_One_image.py        # 1 枚の画像を OCR（結果はターミナルに表示）
├── OCR_multiple_image.py   # フォルダ内の全画像を一括 OCR（TXT/JSONL 出力）
├── requirements.txt
└── README.md
```

---

## 📋 必要要件

- **Python 3.8+**
- **NVIDIA GPU**（VRAM 10GB 以上）— RTX 3080 で動作確認済み
- **CUDA Toolkit** & **cuDNN**

### 動作確認環境

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA RTX 3080 (10GB VRAM) |
| Python | 3.10.11 |
| PyTorch | 2.x + CUDA |
| Model | `tencent/HunyuanOCR`（HuggingFace） |

---

## 🚀 セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/<your-username>/HunyuanOCR_VRAM-10GB.git
cd HunyuanOCR_VRAM-10GB
```

### 2. 依存パッケージをインストール

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate pillow requests
```

> **注意**: 初回実行時に HunyuanOCR モデル（約 5GB）が HuggingFace から自動ダウンロードされます。

---

## ▶️ 使い方

### 1 枚の画像を OCR — `OCR_One_image.py`

スクリプト内の `TARGET_IMAGE` を対象画像のパスに書き換えて実行します。

```python
# OCR_One_image.py 内
TARGET_IMAGE = r"C:\Users\user\Desktop\sample.png"
```

```bash
python OCR_One_image.py
```

結果はターミナルに表示されます:

```
Loading tencent/HunyuanOCR (dtype=torch.bfloat16, eager attention)...
Running OCR inference...
------------------------------
OCR Result:
（認識されたテキストがここに表示）
------------------------------
```

---

### フォルダ内の全画像を一括 OCR — `OCR_multiple_image.py`

```bash
python OCR_multiple_image.py \
    --input_dir "画像フォルダのパス" \
    --glob "review_page_*.png" \
    --output_txt "出力先/output.txt" \
    --output_jsonl "出力先/output.jsonl"
```

#### 引数一覧

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--input_dir` | **必須** | 画像が入っているフォルダのパス |
| `--glob` | `*.png` | 対象ファイルのパターン（例: `review_page_*.png`, `*.jpg`） |
| `--output_txt` | **必須** | まとめ TXT の出力パス |
| `--output_jsonl` | （任意） | JSONL の出力パス |
| `--prompt` | 日本語書き起こし用 | OCR に渡すプロンプト |
| `--max_new_tokens` | `1024` | 生成トークン数の上限（10GB なら 512〜2048 目安） |
| `--long_side` | `1280` | 画像長辺の縮小サイズ（10GB なら 1024〜1600 目安） |

#### 実行例

```bash
# Hokuto の口コミ画像を一括 OCR
python OCR_multiple_image.py \
    --input_dir "./hokuto_reviews" \
    --glob "review_page_*.png" \
    --output_txt "./hokuto_reviews/all_reviews.txt" \
    --output_jsonl "./hokuto_reviews/all_reviews.jsonl"
```

#### 実行ログ

```
Loading tencent/HunyuanOCR (dtype=torch.bfloat16, eager attention)...
[1/47] OCR: ./hokuto_reviews/review_page_001.png
[2/47] OCR: ./hokuto_reviews/review_page_002.png
...
[47/47] OCR: ./hokuto_reviews/review_page_047.png
Saved TXT : ./hokuto_reviews/all_reviews.txt
Saved JSONL: ./hokuto_reviews/all_reviews.jsonl
```

---

## 📄 出力形式

### TXT（ページ区切り付き）

```
===== ./hokuto_reviews/review_page_001.png =====
（1ページ目の認識テキスト）

===== ./hokuto_reviews/review_page_002.png =====
（2ページ目の認識テキスト）

...
```

### JSONL（1 行 = 1 画像）

```json
{"file": "./hokuto_reviews/review_page_001.png", "text": "認識テキスト..."}
{"file": "./hokuto_reviews/review_page_002.png", "text": "認識テキスト..."}
```

---

## ⚙️ VRAM 最適化のポイント

### 画像サイズ (`--long_side`)

| VRAM | 推奨値 | 備考 |
|------|--------|------|
| 10GB | 1024〜1280 | RTX 3080 向け |
| 16GB | 1280〜1600 | RTX 4080 等 |
| 24GB | 1600〜2048 | RTX 3090/4090 |

### トークン数 (`--max_new_tokens`)

| VRAM | 推奨値 | 備考 |
|------|--------|------|
| 10GB | 512〜1024 | 長文の場合は 1024 |
| 16GB+ | 1024〜2048 | |

### OOM 自動リカバリ

CUDA OOM が発生した場合、自動的に以下の設定で 1 回リトライします：
- `long_side` を **256px 縮小**（最小 768px）
- `max_new_tokens` を **半分に縮小**（最小 256）

---

## 🔧 中国語前置きフィルタについて

HunyuanOCR は中国語モデルのため、日本語画像でも先頭に中国語の前置き文を出力することがあります：

```
以下是图片中的文字内容
（ここから本文）
```

`OCR_multiple_image.py` では約 20 パターンの前置き文を正規表現で自動検出・除去し、  
さらに「かな（ひらがな・カタカナ）を含まない短文で中国語キーワードを含む行」も前置きとして除去します。

---

## 🛠️ 技術スタック

| パッケージ | 用途 |
|-----------|------|
| [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | Tencent の VLM ベース OCR モデル |
| [transformers](https://github.com/huggingface/transformers) | モデルのロード・推論 |
| [PyTorch](https://pytorch.org/) | GPU 推論基盤 |
| [Pillow](https://python-pillow.org/) | 画像の読み込み・リサイズ |
| [accelerate](https://github.com/huggingface/accelerate) | `device_map="auto"` によるメモリ最適化 |

---

## 📌 注意事項

- 初回実行時にモデルが **~5GB** ダウンロードされます（`~/.cache/huggingface/` に保存）
- `--long_side` を大きくしすぎると OOM になります。10GB 環境では **1280 以下** を推奨
- URL 画像の読み込みは `OCR_One_image.py` のみ対応（`http://` で始まるパスを指定可能）
- `attn_implementation="eager"` を使用しています（Flash Attention 未使用で互換性重視）

---

## License

MIT
