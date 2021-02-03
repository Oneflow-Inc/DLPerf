#!/bin/bash

DEVICE_ID=${1:-0}
BATCH_SIZE=${2:-128}
DTYPE=${3:-'fp32'}
NUM_STEP=${4:-120}
TEST_NUM=${5:-1}
NODE_NUM=${6:-1}

a=`expr ${#DEVICE_ID} + 1`
GPUS_PER_NODE=`expr ${a} / 2`
total_batch_size=`expr ${BATCH_SIZE} \* $GPUS_PER_NODE`
echo "Use gpus: $DEVICE_ID"
echo "Total batch size : $total_batch_size"

TOTAL_GPU_NUM=`expr ${NODE_NUM} \* ${GPUS_PER_NODE}`
echo "Total use: ${TOTAL_GPU_NUM} gpu"

if [ ${NODE_NUM} -eq 1 ] ; then
  NODE_IP=localhost:${GPUS_PER_NODE}
elif [ ${NODE_NUM} -eq 2 ] ; then
  NODE_IP=${NODE1}:${GPUS_PER_NODE},${NODE2}:${GPUS_PER_NODE}
elif [ ${NODE_NUM} -eq 4 ] ; then
  NODE_IP=${NODE1}:${GPUS_PER_NODE},${NODE2}:${GPUS_PER_NODE},${NODE3}:${GPUS_PER_NODE},${NODE4}:${GPUS_PER_NODE}
else
    echo "Invalid node num."
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export GLOG_logtostderr=1
export GLOG_v=2
LOG_FOLDER=./logs/mindspore/resnet50/bz${BATCH_SIZE}/${NODE_NUM}n${GPUS_PER_NODE}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/rn50_b${BATCH_SIZE}_${DTYPE}_${TEST_NUM}.log

mpirun --allow-run-as-root \
  --prefix /usr/local/openmpi-4.0.3 \
  --output-filename log_output \
  --merge-stderr-to-stdout \
  -n $TOTAL_GPU_NUM -H $NODE_IP \
  -x NCCL_DEBUG=INFO \
  -mca plm_rsh_args "-p ${PORT}" \
  python train.py \
      --net="resnet50" \
      --dataset="imagenet2012" \
      --run_distribute=True \
      --device_target="GPU" \
      --data_sink_steps=10 \
      --train_steps=$NUM_STEP \
      --dataset_path="/workspace/resnet/data/ImageNet/train" \
      --batch_size=$BATCH_SIZE \
      --dtype=$DTYPE \
      --device_num=$GPUS_PER_NODE \
      2>&1 | tee $LOGFILE
