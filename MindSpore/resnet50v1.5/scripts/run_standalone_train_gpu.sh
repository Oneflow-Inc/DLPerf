#!/bin/bash

DEVICE_ID=${1:-0}
BATCH_SIZE=${2:-128}
DTYPE=${3:-'fp32'}
NUM_STEP=${4:-120}
TEST_NUM=${5:-1}

export CUDA_VISIBLE_DEVICES=$DEVICE_ID

export GLOG_logtostderr=1
export GLOG_v=2
LOG_FOLDER=./logs/mindspore/resnet50/bz${BATCH_SIZE}/1n1g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/rn50_b${BATCH_SIZE}_${DTYPE}_${TEST_NUM}.log

python train.py \
    --net="resnet50" \
    --dataset="imagenet2012" \
    --device_target="GPU" \
    --data_sink_steps=10 \
    --train_steps=$NUM_STEP \
    --dataset_path="/workspace/resnet/data/ImageNet/train" \
    --batch_size=$BATCH_SIZE \
    --dtype=$DTYPE \
    2>&1 | tee $LOGFILE
