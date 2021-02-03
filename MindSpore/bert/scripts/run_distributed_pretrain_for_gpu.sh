#!/bin/bash

DEVICE_ID=${1:-0}
BATCH_SIZE=${2:-32}
DTYPE=${3:-'fp32'}
NUM_STEP=${4:-120}
ENABLE_GRAPH_KERNEL=${5:-'false'}
TEST_NUM=${6:-1}
NODE_NUM=${7:-1}

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

ENABLE_LOSSSCALE="false"
if [ ${DTYPE} == "fp16" ] ; then
  ENABLE_LOSSSCALE="true"
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export GLOG_logtostderr=1
export GLOG_v=2
LOG_FOLDER=./logs/mindspore/bert/bz${BATCH_SIZE}/${NODE_NUM}n${GPUS_PER_NODE}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/bert_b${BATCH_SIZE}_${DTYPE}_${TEST_NUM}.log

mpirun --allow-run-as-root \
  --prefix /usr/local/openmpi-4.0.3 \
  --output-filename log_output \
  --merge-stderr-to-stdout \
  -n $TOTAL_GPU_NUM -H $NODE_IP \
  -x NCCL_DEBUG=INFO \
  -mca plm_rsh_args "-p ${PORT}" \
  python run_pretrain.py        \
    --device_target="GPU"      \
    --distribute="true"        \
    --epoch_size=1    \
    --enable_save_ckpt="false"    \
    --enable_lossscale=$ENABLE_LOSSSCALE \
    --enable_data_sink="true"    \
    --data_sink_steps=10        \
    --train_steps=$NUM_STEP \
    --data_dir="/workspace/bert/data/wiki" \
    --enable_graph_kernel=$ENABLE_GRAPH_KERNEL \
    --batch_size=$BATCH_SIZE \
    --dtype=$DTYPE \
    --schema_dir="" 2>&1 | tee $LOGFILE

