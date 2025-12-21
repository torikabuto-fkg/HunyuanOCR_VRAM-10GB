# HunyuanOCR_VRAM-10GB
Description: RTX 3080 (VRAM 10GB) 環境で HunyuanOCR を動作させるための軽量化・最適化リポジトリ。 (Optimized HunyuanOCR repository for running on RTX 3080 with 10GB VRAM.)

# OCR_One_image.py
一枚の画像ファイルのみOCR可能。テキストデータはターミナルに表示されます。

python OCR_One_image.py
で実行可

# OCR_multiple_image.py
フォルダ内のすべての画像を（数字の）順にOCR可能。テキストデータは画像の入っているフォルダに .txtファイルと、 .jsonlファイルで出力されます。

python hunyuan_batch_ocr.py   --input_dir "＄データの入っているフォルダのPATH＄"   --glob "review_page_*.png（フォルダの画像の名。今回はreview_page_*.pngのアスタリスクの部分が数字で順になっている）"   --output_txt "＄アウトプットディレクトリのPATH＄.txt"   --output_jsonl "＄アウトプットディレクトリのPATH＄.jsonl"
