#! /bin/bash
MODEL=${1:-gpt2-small}
BATCH_SIZE_PER_DEVICE=${2:-8}
NUM_WORKERS=${3:-1}
NUM_GPUS_PER_WORKER=${4:-8}
ZERO_STAGE=${5:-2}
CHECKPOINT_ACTIVATIONS=${6:-"on"}
DTYPE=${7:-'fp16'}
TEST_NUM=${8:-1}
ITER_NUM=${9:-200}
MP_SIZE=${10:-1}

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

a=`expr ${#GPUS} + 1`
gpu_num_per_node=`expr ${a} / 2`
gpu_num=`expr ${NUM_GPUS_PER_WORKER} \* ${NUM_WORKERS}`
total_bz=`expr ${BATCH_SIZE_PER_DEVICE} \* ${gpu_num}`

sed -i "s/\"train_batch_size\":.*$/\"train_batch_size\": $total_bz,/"   $script_dir/ds_zero2_config.json
if  [ ${CHECKPOINT_ACTIVATIONS} == "on" ];then
    sed -i "s/\"partition_activations\":.*$/\"partition_activations\": true,/"   $script_dir/ds_zero2_config.json
else
    sed -i "s/\"partition_activations\":.*$/\"partition_activations\": false,/"   $script_dir/ds_zero2_config.json
fi
sed -i "s/\"stage\":.*$/\"stage\": $ZERO_STAGE/"   $script_dir/ds_zero2_config.json


if  [ ${MODEL} == "gpt2-small" ];then
    echo "Using network >> gpt2-small"
    num_layers=12
    num_attention_heads=12
    hidden_size=768
elif  [ ${MODEL} == "gpt2-medium" ];then
    echo "Using network >> gpt2-medium"
    num_layers=24
    num_attention_heads=16
    hidden_size=1024
fi

PREFIX=logs-20210414-stage${ZERO_STAGE}-${CHECKPOINT_ACTIVATIONS}-activation
rm -rf test-checkpoints
LOG_FOLDER=./${PREFIX}/deepspeed/${MODEL}/bz${BATCH_SIZE_PER_DEVICE}/${NUM_WORKERS}n${NUM_GPUS_PER_WORKER}g
mkdir -p $LOG_FOLDER
LOG=${LOG_FOLDER}/${MODEL}_b${BATCH_SIZE_PER_DEVICE}_fp16_${TEST_NUM}.log


config_json="$script_dir/ds_zero2_config.json"
gpt_options=" \
      --save test-checkpoints  \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${num_layers} \
       --hidden-size ${hidden_size} \
       --num-attention-heads ${num_attention_heads} \
       --batch-size ${BATCH_SIZE_PER_DEVICE} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters ${ITER_NUM}  \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
"

if [ "$DTYPE" = "fp16" ] ; then
    echo "Using data type >> fp16"
    gpt_options="${gpt_options} --fp16 "
else
    echo "Using data type >> fp32"
fi


if  [ ${CHECKPOINT_ACTIVATIONS} == "on" ];then
    gpt_options="${gpt_options}   
              --checkpoint-activations   --deepspeed-activation-checkpointing    --deepspeed   --deepspeed_config ${config_json} "
else
    gpt_options="${gpt_options} 
               --deepspeed \
               --deepspeed_config ${config_json} \
    "
fi

run_cmd="deepspeed   --hostfile=deepspeed_hosts   --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_gpt2.py  ${gpt_options} "
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee ${LOG}
