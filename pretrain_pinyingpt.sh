#!/usr/bin/env bash
# Copyright (c) VisualJoyce.
# Licensed under the MIT license.

WORK_DIR=$(dirname $(readlink -f $0))
echo "${WORK_DIR}"

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth1
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
unset OMPI_MCA_plm_rsh_agent

echo "Training using ${NODE_NUM} GPUs!"
echo "NODE NUM: ${NODE_NUM}"
echo "NODE LIST: ${NODE_LIST}"
echo "NODE IP LIST: ${NODE_IP_LIST}"
echo "HOST_GPU_NUM: ${HOST_GPU_NUM}"
echo "HOST_NUM: ${HOST_NUM}"
echo "INDEX: ${INDEX}"
echo "CHIEF_IP: ${CHIEF_IP}"

MODELS_DIR="${OUTPUT_DIR}"/"${TJ_TASK_ID}"/"${TJ_INSTANCE_ID}"/models
LOGGING_DIR="${OUTPUT_DIR}"/"${TJ_TASK_ID}"/"${TJ_INSTANCE_ID}"/logs
mkdir -p "$LOGGING_DIR"

torchrun --nproc_per_node="$HOST_GPU_NUM" --nnodes="$HOST_NUM" --node_rank="$INDEX" \
  --master_addr="$CHIEF_IP" --master_port=23456 \
  "${WORK_DIR}"/"${PROJECT}"_pretrain.py \
  --train_text_pinyin_files /jizhi_data/minghuantan/corpus/tcorpus \
  --model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
  --pinyin_logits_processor_cls pinyingpt-concat \
  --abbr_mode full \
  --fp16 \
  --do_train \
  --do_eval \
  --text_pinyin_per_device_train_batch_size 64 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate 5e-5 --weight_decay 0.01 \
  --warmup_steps 1000 \
  --save_steps 1000 \
  --logging_steps 1000 \
  --text_only_block_size 128 \
  --output_dir "${MODELS_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  --log_on_each_node false \
  --use_at_most_k "${BATCHES_USED_PER_MODALITY}" \
  --max_steps 100000 > "$LOGGING_DIR"/log.txt
