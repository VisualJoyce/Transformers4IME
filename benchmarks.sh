#!/usr/bin/env bash
shopt -s nullglob

MODEL_NAME=$1
MODEL_CKPT=$2
ABBR_MODE=$3
GPU_CARD=$4
NUM_BEAMS=16

if [ -z "$GPU_CARD" ]; then
  GPU_CARD=0
fi

DOMAINS=(
  "医学问答"
  "体育"
  "军事"
  "农业"
  "国际"
  "娱乐"
  "房产"
  "文化"
  "教育"
  "旅行"
  "汽车"
  "游戏"
  "社会"
  "科技"
  "经济"
  "财经"
)

SAMPLES_JSON_DIRS=(
  "_0-3_0-3"
  "_0-3_10+"
  "_0-3_4-9"
  "_10+_0-3"
  "_10+_10+"
  "_10+_4-9"
  "_4-9_0-3"
  "_4-9_10+"
  "_4-9_4-9"
)

echo $MODEL_CKPT
MODEL_DIR=$(dirname "$MODEL_CKPT")
MODEL_DIR=$(dirname "$MODEL_DIR")
cur_ckpt=$(basename "$MODEL_CKPT")
cur_ckpt=${cur_ckpt##model_step_}
cur_ckpt=${cur_ckpt%%.pt}

for DOMAIN in "${DOMAINS[@]}"; do
  for SUB in "${SAMPLES_JSON_DIRS[@]}"; do
    SAMPLES_JSON="/apdcephfs/share_916081/minghuantan/p2z/raw/wudao/samples_${DOMAIN}${SUB}.json"
    echo "${SAMPLES_JSON}"
    PYTHONPATH=src python benchmarks.py \
      --pretrained_model_name_or_path=/apdcephfs/share_916081/minghuantan/pretrained_models/pinyingpt \
      --abbr_mode="${ABBR_MODE}" --samples_json "${SAMPLES_JSON}" \
      --best_pt "$MODEL_CKPT" \
      --model_name "${MODEL_NAME}" \
      --output_dir "$MODEL_DIR/log" \
      --pinyin_logits_processor_cls pinyingpt-concat \
      --num_beams "${NUM_BEAMS}" --device "${GPU_CARD}" --global_step "$cur_ckpt"
  done
done
