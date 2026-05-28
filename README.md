# HunyuanOCR & Qwen2.5-VL-7B-Instruct-AWQ Batch OCR Toolkit (for VRAM 10GB)

このリポジトリは、VRAMが10GB前後のGPU（NVIDIA RTX 3080、RTX A4000など）環境に最適化された、高精度マルチモーダルVLMによるバッチ（一括）OCR処理用ツールキットです。

高性能な **Tencent HunyuanOCR** と **Qwen2.5-VL-7B-Instruct-AWQ** (4bit量子化版) の双方に対応し、OOM自動リカバリや余分な中国語の前置き文の自動クレンジング機能などを備えています。

---

## 📋 必要要件

- **Python 3.8+ (3.10推奨)**
- **NVIDIA GPU**（VRAM 10GB 以上）— RTX 3080、RTX A4000 で動作確認済み
- **CUDA Toolkit** & **cuDNN**

### 動作確認環境

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA RTX 3080 (10GB VRAM) / RTX A4000 (16GB VRAM) |
| Python | 3.10.x |
| PyTorch | 2.6.0 + CUDA 12.4 |
| Models | `tencent/HunyuanOCR` / `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` |

---

## 🚀 セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/<your-username>/HunyuanOCR_VRAM-10GB.git
cd HunyuanOCR_VRAM-10GB
```

### 2. 仮想環境の構築と依存パッケージのインストール

HunyuanOCR は比較的新しいモデルであり、公式の推奨ブランチから `transformers` をインストールする必要があります。

```bash
# 仮想環境作成 (推奨)
conda create -n hunyuan-env python=3.10 -y
conda activate hunyuan-env

# PyTorch インストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# HunyuanOCR対応の特定 transformers コミットをインストール
pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4

# その他の依存パッケージをインストール
pip install pillow requests accelerate autoawq qwen-vl-utils huggingface-hub
```

> **注意**: 初回実行時に、指定したモデルのチェックポイント（HunyuanOCR: 約5GB / Qwen2.5-VL-AWQ: 約5GB）が HuggingFace Hub から自動的にダウンロードされます。

---

## 🛠️ トラブルシューティング（非常に重要）

環境構築時、およびスクリプトの実行時には以下の2つのエラーが発生することがあります。本ツールでは、それらに対して既に対策コードを組み込み、対処方法を整理しています。

### ① OSError: tencent/HunyuanOCR does not appear to have a file named modeling_hunyuan_ocr.py
- **原因**: ネット上の古い記事にある `get_class_from_dynamic_module` を使ったハックコードは、Hugging Face 上の `tencent/HunyuanOCR` にモデル定義の `.py` ファイルが含まれていないため動作しません。
- **対策**: `transformers` の最新対応版に内蔵されている正規のクラス `HunYuanVLForConditionalGeneration` をインポートしてロードすることで、ハックを排して完全に解決しています。

### ② ImportError: cannot import name 'PytorchGELUTanh' from 'transformers.activations'
- **原因**: 特定コミット版の `transformers` では `PytorchGELUTanh` が `GELUTanh` にリネーム（あるいは削除）されており、`autoawq` のインポート時にエラーが発生して Qwen の AWQ モデルが読み込めなくなります。
- **対策**: 環境内の `autoawq` ライブラリのファイルを以下のように修正します。

**修正対象ファイル**:
`~/miniconda3/envs/hunyuan-env/lib/python3.10/site-packages/awq/quantize/scale.py`

**修正内容 (12行目付近のインポート処理を try-except でラップ)**:
```python
# 修正前：
# from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation

# 修正後：
try:
    from transformers.activations import NewGELUActivation, GELUActivation, PytorchGELUTanh
except ImportError:
    from transformers.activations import NewGELUActivation, GELUActivation
    try:
        from transformers.activations import GELUTanh as PytorchGELUTanh
    except ImportError:
        PytorchGELUTanh = None
```
および、その直後の `allowed_act_fns`（16行目付近）のリストから `PytorchGELUTanh` を除外し、最後に条件付きで追加します：
```python
allowed_act_fns = [
    nn.GELU,
    BloomGelu,
    NewGELUActivation,
    GELUActivation,
]
if PytorchGELUTanh is not None:
    allowed_act_fns.append(PytorchGELUTanh)
```

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

結果はターミナルに表示されます。

---

### フォルダ内の全画像を一括 OCR (バッチモード)

本リポジトリのメインスクリプトは、コマンドライン引数（`argparse`）に完全対応し、引数を変えるだけで多様な設定で一括処理可能です。自然順ソート（001 -> 002 -> 010...）されて順次処理されます。

#### Tencent HunyuanOCR を使う場合: `OCR_multiple_image.py`
```bash
python OCR_multiple_image.py \
    --input_dir "./画像" \
    --glob "review_page_*.png" \
    --output_txt "./output_hunyuan.txt" \
    --output_jsonl "./output_hunyuan.jsonl"
```

#### Qwen2.5-VL-7B-Instruct-AWQ を使う場合: `run_qwen.py`
```bash
python run_qwen.py \
    --input_dir "./画像" \
    --glob "review_page_*.png" \
    --output_txt "./output_qwen.txt" \
    --output_jsonl "./output_qwen.jsonl"
```

#### 引数一覧 (両スクリプト共通)

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--input_dir` | **必須** | 画像が入っているフォルダのパス |
| `--glob` | `*.png` | 対象ファイルのパターン（例: `review_page_*.png`, `*.jpg`） |
| `--output_txt` | **必須** | まとめ TXT の出力パス |
| `--output_jsonl` | （任意） | JSONL の出力パス |
| `--prompt` | 日本語書き起こし用 | OCR に渡すプロンプト |
| `--max_new_tokens` | `1024` | 生成トークン数の上限（10GB なら 512〜2048 目安） |
| `--long_side` | `1280` | [HunyuanOCRのみ] 画像長辺の縮小サイズ（10GB なら 1024〜1280 推奨） |

---

## 📄 出力形式

### TXT（ページ区切り付き）

```
===== ./hokuto_reviews/review_page_001.png =====
（1ページ目の認識テキスト）

===== ./hokuto_reviews/review_page_002.png =====
（2ページ目の認識テキスト）
```

### JSONL（1 行 = 1 画像）

```json
{"file": "./hokuto_reviews/review_page_001.png", "text": "認識テキスト..."}
{"file": "./hokuto_reviews/review_page_002.png", "text": "認識テキスト..."}
```

---

## ⚙️ VRAM 最適化と OOM 自動リカバリ

### 画像サイズとVRAM目安

| VRAM | HunyuanOCR 推奨画像サイズ (`--long_side`) | Qwen2.5-VL 最大ピクセル設定 | 備考 |
|------|-----------------------|-----------------------|------|
| 10GB | 1024〜1280 | 768 * 768 制限 (内蔵済) | RTX 3080 / RTX 4070 向け |
| 16GB | 1280〜1600 | 1024 * 1024 制限 | RTX 4080 / RTX A4000 等 |
| 24GB | 1600〜2048 | 制限なし（自動） | RTX 3090 / RTX 4090 等 |

### 🛡️ OOM 自動リカバリ機能搭載
推論中に CUDA Out Of Memory (OOM) が発生した場合、スクリプトがクラッシュするのを防ぐために**自動的に1回のみパラメータを自動縮小して再試行**するリカバリ機能が組み込まれています。
- **HunyuanOCR**: `--long_side` を **256px 縮小**（最小 768px）、`max_new_tokens` を**半分**にして再実行。
- **Qwen2.5-VL**: `max_pixels` を**半分**に削減、`max_new_tokens` を**半分**にして再実行。

---

## 🧹 中国語前置きの自動フィルタ機能について

HunyuanOCR や Qwen はマルチリンガル/中国語ベースのモデルであるため、日本語の画像を処理する場合でも、稀にテキストの先頭に以下のような余計な中国語の前置き文を出力することがあります。

> 「以下是图片中的文字内容：」 (以下は画像内の文字内容です)

本リポジトリのバッチ処理スクリプトには、これらの代表的な前置き文約 20 パターンを自動で検知して削り落とす**クレンジングフィルター（`clean_ocr_text`）**が組み込まれています。また、「かな（ひらがな・カタカナ）を含まない短文で中国語キーワードを含む行」も検知して自動除去されるため、出力ファイルにはクリアな日本語テキストだけが保存されます。

---

## 🛠️ 技術スタック

| パッケージ | 用途 |
|-----------|------|
| [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | Tencent の VLM ベース超高精度 OCR モデル |
| [Qwen2.5-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ) | 4bit量子化された最高峰のオープンソース VLM モデル |
| [transformers](https://github.com/huggingface/transformers) | モデルのロード・推論フレームワーク |
| [PyTorch](https://pytorch.org/) | GPU 推論基盤 |
| [Pillow](https://python-pillow.org/) | 画像の読み込み・最適化リサイズ |
| [accelerate](https://github.com/huggingface/accelerate) | `device_map="auto"` によるメモリ最適化 |

---

## License

MIT
