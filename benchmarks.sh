#!/usr/bin/env bash
shopt -s nullglob
set -x

# pinyingpt-concat
# pinyingpt-compatible
EVAL_TYPE=$1
OUTPUT_DIR=$2
BEST_PT=$3
if [ -z "$BEST_PT" ]; then
  PYTHONPATH=src python3 benchmarks.py \
    --samples_json_dir /apdcephfs/share_916081/minghuantan/p2z/raw/wudao \
    --pretrained_model_name_or_path /apdcephfs/share_916081/minghuantan/pretrained_models/gpt2-zh-ours \
    --additional_special_tokens /apdcephfs/share_916081/minghuantan/p2z/pretrained/additional_special_tokens.json \
    --pinyin2char_json /apdcephfs/share_916081/minghuantan/p2z/pretrained/pinyin2char.json \
    --model_name "$EVAL_TYPE" \
    --pinyin_logits_processor_cls "$EVAL_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --num_beams 16 \
    --abbr_mode full
else
  PYTHONPATH=src python3 benchmarks.py \
    --samples_json_dir /apdcephfs/share_916081/minghuantan/p2z/raw/wudao \
    --pretrained_model_name_or_path /apdcephfs/share_916081/minghuantan/pretrained_models/gpt2-zh-ours \
    --best_pt "$BEST_PT" \
    --additional_special_tokens /apdcephfs/share_916081/minghuantan/p2z/pretrained/additional_special_tokens.json \
    --pinyin2char_json /apdcephfs/share_916081/minghuantan/p2z/pretrained/pinyin2char.json \
    --model_name "$EVAL_TYPE" \
    --pinyin_logits_processor_cls "$EVAL_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --num_beams 16 \
    --abbr_mode full
fi

