#!/bin/bash
OUTPUT_DIR=../output
# Where should we save checkpoints and tensorboard events?
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
MODEL=${1:-"bert_base"}
BATCH_SIZE=${2:-32}
gpus=${3:-"0"}
nodes=${4:-$NODE1}
TEST_NUM=${5:-1}
DTYPE=${6:-"fp32"}

a=`expr ${#gpus} + 1`
num_gpus=`expr ${a} / 2`
num_nodes=$(echo $nodes | tr ',' '\n' | wc -l)
train_batch_size=`expr ${BATCH_SIZE} \* 1024`

LOG_FOLDER=../logs-${DTYPE}/deepspeed/bert/bz${BATCH_SIZE}/${num_nodes}n${num_gpus}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/bert_b${BATCH_SIZE}_${DTYPE}_$TEST_NUM.log

job_name=adam_nvidia_data_${MODEL}
config=${MODEL}.json
deepspeed_config=deepspeed_bsz64k_adam_config_seq128.json
# deepspeed_config=deepspeed_bsz4k_onebit_config_seq128.json

if  [ ${DTYPE} == "fp16" ];then
    enabled=true
else
    enabled=false
fi

sed -i "s/\"train_batch_size\":.*$/\"train_batch_size\": $train_batch_size,/" $deepspeed_config
sed -i "s/\"train_micro_batch_size_per_gpu\":.*$/\"train_micro_batch_size_per_gpu\": $BATCH_SIZE,/" $deepspeed_config
sed -i "s/\"enabled\":.*$/\"enabled\":$enabled,/" $deepspeed_config


DATA_PATH_PREFIX=/datasets/bert/deepspeed/data/test
if  [ $num_nodes -ge 2 ];then
    NCCL_TREE_THRESHOLD=0 deepspeed   --hostfile=deepspeed_hosts \
        --num_nodes=$num_nodes \
        --num_gpus=$num_gpus  deepspeed_train.py \
        --cf  ${config}  \
        --max_seq_length 128 \
        --output_dir $OUTPUT_DIR \
        --deepspeed \
        --print_steps 1 \
        --lr_schedule "EP" \
        --max_steps_per_epoch 120 \
        --lr_offset 10e-4 \
        --job_name ${job_name} \
        --deepspeed_config  $deepspeed_config \
        --data_path_prefix  ${DATA_PATH_PREFIX}   \
        --use_nvidia_dataset  2>&1 | tee $LOGFILE
else
    NCCL_TREE_THRESHOLD=0 deepspeed  \
        --num_nodes=$num_nodes \
        --num_gpus=$num_gpus  deepspeed_train.py \
        --cf  ${config}  \
        --max_seq_length 128 \
        --output_dir $OUTPUT_DIR \
        --deepspeed \
        --print_steps 1 \
        --lr_schedule "EP" \
        --max_steps_per_epoch 120 \
        --lr_offset 10e-4 \
        --job_name ${job_name} \
        --deepspeed_config  $deepspeed_config  \
        --data_path_prefix  ${DATA_PATH_PREFIX}   \
        --use_nvidia_dataset  2>&1 | tee $LOGFILE
fi

