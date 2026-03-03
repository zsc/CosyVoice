#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
set -euo pipefail
. ./path.sh || exit 1;

# Stages:
#   0: Prepare wav.scp/text/utt2spk/spk2utt from wav+lab dataset tree
#   1: Extract campplus speaker embedding
#   2: Extract discrete speech token
#   3: Make parquet shards + data.list
#   4: Generate train.data.list/dev.data.list
#   5: Train llm/flow (CosyVoice-300M)

stage=5
stop_stage=5

src_data_dir="/mnt/sda2/中文 - Chinese"
speaker="派蒙"
data_dir="/mnt/sda2/cosyvoice_data/paimon"
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M
train_models="${train_models:-llm flow}"
train_config="${train_config:-conf/cosyvoice.yaml}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation (wav+lab -> wav.scp/text/utt2spk/spk2utt)"
  python ../../../tools/prepare_wav_lab_dataset.py \
    --src_dir "${src_data_dir}" \
    --des_dir "${data_dir}" \
    --speakers "${speaker}" \
    --dev_ratio 0.02
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding"
  for x in train dev; do
    ../../../tools/extract_embedding.py --dir "${data_dir}/${x}" \
      --onnx_path "${pretrained_model_dir}/campplus.onnx"
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token"
  for x in train dev; do
    ../../../tools/extract_speech_token.py --dir "${data_dir}/${x}" \
      --onnx_path "${pretrained_model_dir}/speech_tokenizer_v1.onnx"
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Make parquet shards"
  for x in train dev; do
    mkdir -p "${data_dir}/${x}/parquet"
    ../../../tools/make_parquet_list.py --num_utts_per_parquet 200 \
      --num_processes 10 \
      --src_dir "${data_dir}/${x}" \
      --des_dir "${data_dir}/${x}/parquet"
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Generate train.data.list/dev.data.list"
  cp "${data_dir}/train/parquet/data.list" "${data_dir}/train.data.list"
  cp "${data_dir}/dev/parquet/data.list" "${data_dir}/dev.data.list"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=50
train_engine=torch_ddp

exp_root="${exp_root:-/mnt/sda2/cosyvoice_exp/paimon}"
tensorboard_root="${tensorboard_root:-/mnt/sda2/cosyvoice_tensorboard/paimon}"
log_dir="${log_dir:-/mnt/sda2/cosyvoice_exp/paimon/logs}"
mkdir -p "${log_dir}"
# Reduce CUDA allocator fragmentation on long runs (helps small VRAM GPUs).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run training (CosyVoice-300M SFT): llm + flow"
  for model in ${train_models}; do
    log_path="${log_dir}/${model}_$(date +%Y-%m-%d_%H%M%S).log"
    echo "Logging to ${log_path}"
    checkpoint_path="${pretrained_model_dir}/${model}.pt"
    if [ -n "${checkpoint_override:-}" ]; then
      checkpoint_path="${checkpoint_override}"
    fi
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
      ../../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config "${train_config}" \
      --train_data "${data_dir}/train.data.list" \
      --cv_data "${data_dir}/dev.data.list" \
      --model $model \
      --checkpoint "${checkpoint_path}" \
      --model_dir "${exp_root}/${model}/${train_engine}" \
      --tensorboard_dir "${tensorboard_root}/${model}/${train_engine}" \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp 2>&1 | tee "${log_path}"
    status=$?
    if [ $status -ne 0 ]; then
      echo "Training failed for model=$model (exit=$status). Stop." >&2
      exit $status
    fi
  done
fi
