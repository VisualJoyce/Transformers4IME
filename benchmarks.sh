#!/usr/bin/env bash
shopt -s nullglob

PYTHONPATH=src python3 benchmarks.py \
  --samples_json_dir /apdcephfs/share_916081/minghuantan/p2z/raw/wudao \
  --pretrained_model_name_or_path /apdcephfs/share_916081/minghuantan/pretrained_models/gpt2-zh-ours \
  --best_pt "$1" \
  --additional_special_tokens /apdcephfs/share_916081/minghuantan/p2z/pretrained/additional_special_tokens.json \
  --pinyin2char_json /apdcephfs/share_916081/minghuantan/p2z/pretrained/pinyin2char.json \
  --model_name pinyingpt-concat \
  --pinyin_logits_processor_cls pinyingpt-concat \
  --num_beams 16 \
  --abbr_mode full
