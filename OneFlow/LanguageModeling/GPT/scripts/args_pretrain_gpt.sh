#! /bin/bash

# Runs the "117M" parameter model

#BASEDIR="$(dirname $0)"
BASE_DIR=/data/OneFlow-Benchmark/LanguageModeling/GPT
DATA_PATH=/data/gpt/gpt_sample_dataset_text_document
CHECKPOINT_PATH=${BASE_DIR}/model_save
rm -rf ${BASE_DIR}/model_save/*
export PYTHONUNBUFFERED=1
export PYTHONPATH=$BASE_DIR/..:$PYTHONPATH
#export ONEFLOW_DEBUG_MODE=1
export NCCL_DEBUG=INFO

M_P=${1:-1}
P_P=${2:-1}
MICRO_BATCH_SIZE=${3:-8}
GLOABAL_BATCH_SIZE=${4:-16}
NNODES=${5:-1}
GPUS_PER_NODE=${6:-8}
TRAIN_ITERS=${7:-500}

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

D_P=$(($WORLD_SIZE/$M_P/$P_P))

NUM_LAYERS=${8:-16}
HIDDEN_SIZE=${9:-1536}
NUM_ATTENTION_HEADS=${10:-16}
SEQ_LENGTH=2048
DROPOUT_RATE=0.1

LOGFILE=${BASE_DIR}/oneflow_gpt_${NNODES}n${GPUS_PER_NODE}g_dp${D_P}_mp${M_P}_pp${P_P}_mbs${MICRO_BATCH_SIZE}_gbs${GLOABAL_BATCH_SIZE}_l${NUM_LAYERS}_hsz${HIDDEN_SIZE}_ahs${NUM_ATTENTION_HEADS}_repeat_pretrain.log

python3 -m oneflow_gpt.training \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOABAL_BATCH_SIZE \
    --tensor-model-parallel-size $M_P \
    --pipeline-model-parallel-size $P_P \
    --num-gpus-per-node $GPUS_PER_NODE \
    --train-iters $TRAIN_ITERS \
    --learning-rate 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 320000 \
    --lr-warmup-fraction 0.01 \
    --optimizer adamw \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --dataset $DATA_PATH \
    --seq-length $SEQ_LENGTH \
    --hidden-dropout $DROPOUT_RATE \
    --attention-dropout $DROPOUT_RATE \
    --vocab-size 50257 \
    --split 949,50,1 \
    --save $CHECKPOINT_PATH \
    --save-interval 10000 \
    --log-interval 10 \
    --metric-print-format table \
    --checkpoint-activations \
    --multihead-attention-fusion \
    --num-nodes $NNODES \
    --node-ips 10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5 \
    --use-rdma \
    --fp16 | tee $LOGFILE
