#!/bin/bash

DEVICE_ID=${1:-0}
BATCH_SIZE=${2:-32}
DTYPE=${3:-'fp32'}
NUM_STEP=${4:-120}
TEST_NUM=${5:-1}

a=`expr ${#DEVICE_ID} + 1`
NUM_GPU=`expr ${a} / 2`
total_batch_size=`expr ${BATCH_SIZE} \* $NUM_GPU`
echo "Use gpus: $DEVICE_ID"
echo "Total batch size : $total_batch_size"
RANK_SIZE=$NUM_GPU

export CUDA_VISIBLE_DEVICES=$DEVICE_ID

export GLOG_logtostderr=1
export GLOG_v=2
LOG_FOLDER=./logs/mindspore/bert/bz${BATCH_SIZE}/1n${NUM_GPU}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/bert_b${BATCH_SIZE}_${DTYPE}_${TEST_NUM}.log

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  python run_pretrain.py        \
    --device_target="GPU"      \
    --distribute="true"        \
    --epoch_size=1    \
    --enable_save_ckpt="false"    \
    --enable_lossscale="false"    \
    --enable_data_sink="true"    \
    --data_sink_steps=10        \
    --train_steps=120 \
    --data_dir="/workspace/bert/data/wiki" \
    --batch_size=$BATCH_SIZE \
    --dtype=$DTYPE \
    --schema_dir="" 2>&1 | tee $LOGFILE

