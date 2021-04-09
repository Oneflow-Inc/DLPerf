#! /bin/bash

# Change for multinode config
BATCH_SIZE=${1:-4}
NUM_GPUS_PER_WORKER=${2:-8}
ZERO_STAGE=${3:-0}
CHECKPOINT_ACTIVATIONS=${4:-"off"}
NUM_WORKERS=${5:-1}
MP_SIZE=${6:-1}
ITER_NUM=${7:-1000}

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

echo "BATCH_SIZE: ${BATCH_SIZE}, NUM_GPUS_PER_WORKER:${NUM_GPUS_PER_WORKER},  ZERO_STAGE:${ZERO_STAGE}, CHECKPOINT_ACTIVATIONS:${CHECKPOINT_ACTIVATIONS} "

a=`expr ${#GPUS} + 1`
gpu_num_per_node=`expr ${a} / 2`
gpu_num=`expr ${NUM_GPUS_PER_WORKER} \* ${NUM_WORKERS}`
total_bz=`expr ${BATCH_SIZE} \* ${gpu_num}`

sed -i "s/\"train_batch_size\":.*$/\"train_batch_size\": $total_bz,/"   $script_dir/ds_zero2_config.json
if  [ ${CHECKPOINT_ACTIVATIONS} == "on" ];then
    sed -i "s/\"partition_activations\":.*$/\"partition_activations\": true,/"   $script_dir/ds_zero2_config.json
else
    sed -i "s/\"partition_activations\":.*$/\"partition_activations\": false,/"   $script_dir/ds_zero2_config.json
fi
sed -i "s/\"stage\":.*$/\"stage\": $ZERO_STAGE/"   $script_dir/ds_zero2_config.json

# gpt2-small
num_layers=12
num_attention_heads=12
hidden_size=768

# # gpt2-medium
# num_layers=24
# num_attention_heads=16
# hidden_size=1024


PREFIX=20201209-test_zero_gpt2-small
rm -rf checkpoints
LOG_FOLDER=./logs
mkdir -p $LOG_FOLDER
LOG=${LOG_FOLDER}/${PREFIX}_${NUM_WORKERS}n${NUM_GPUS_PER_WORKER}g_bz${BATCH_SIZE}_zero_stage${ZERO_STAGE}_${CHECKPOINT_ACTIVATIONS}_checkpoint_activation.log



config_json="$script_dir/ds_zero2_config.json"
gpt_options=" \
      --save $PREFIX_checkpoint_${NUM_WORKERS}n${NUM_GPUS_PER_WORKER}g_bz${BATCH_SIZE}_zero_stage${ZERO_STAGE}_${CHECKPOINT_ACTIVATIONS}_checkpoint_activation  \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${num_layers} \
       --hidden-size ${hidden_size} \
       --num-attention-heads ${num_attention_heads} \
       --batch-size ${BATCH_SIZE} \
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
       --fp16 \
"

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