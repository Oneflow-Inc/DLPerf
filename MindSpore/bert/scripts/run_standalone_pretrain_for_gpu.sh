#!/bin/bash

DEVICE_ID=${1:-0}
BATCH_SIZE=${2:-32}
DTYPE=${3:-'fp32'}
NUM_STEP=${4:-120}
ENABLE_GRAPH_KERNEL=${5:-'false'}
TEST_NUM=${6:-1}

export CUDA_VISIBLE_DEVICES=$DEVICE_ID

export GLOG_logtostderr=1
export GLOG_v=2
LOG_FOLDER=./logs/mindspore/bert/bz${BATCH_SIZE}/1n1g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/bert_b${BATCH_SIZE}_${DTYPE}_${TEST_NUM}.log

python run_pretrain.py  \
    --device_target="GPU" \
    --distribute="false" \
    --epoch_size=1 \
    --enable_save_ckpt="false" \
    --enable_lossscale="false" \
    --enable_data_sink="true" \
    --data_sink_steps=10 \
    --train_steps=$NUM_STEP \
    --data_dir="/workspace/bert/data/wiki" \
    --enable_graph_kernel=$ENABLE_GRAPH_KERNEL \
    --batch_size=$BATCH_SIZE \
    --dtype=$DTYPE \
    --schema_dir="" 2>&1 | tee $LOGFILE

