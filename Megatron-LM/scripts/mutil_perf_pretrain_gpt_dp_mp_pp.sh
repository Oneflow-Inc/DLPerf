#! /bin/bash

export NCCL_SOCKET_IFNAME=ib0

export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/data/Megatron-LM
DATA_PATH=/data/gpt/gpt_sample_dataset_text_document
CHECKPOINT_PATH=/data/perf_output
rm -rf /data/perf_output/*
M_P=${1:-1}
P_P=${2:-1}
MICRO_BATCH_SIZE=${3:-8}
GLOABAL_BATCH_SIZE=${4:-16}
NNODES=${5:-1}
GPUS_PER_NODE=${6:-8}
MASTER_ADDR=${7:-127.0.0.1}
MASTER_PORT=21327
NODE_RANK=${8:-0}
echo NODE_RANK=$NODE_RANK
TRAIN_ITERS=${9:-520}

NUM_LAYERS=${10:-16}
HIDDEN_SIZE=${11:-1536}
NUM_ATTENTION_HEADS=${12:-16}
SEQ_LENGTH=2048
DROPOUT_RATE=0.1

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

D_P=$(($WORLD_SIZE/$M_P/$P_P))

LOGFILE=./megatron_lm_perf_${NNODES}n${GPUS_PER_NODE}g_dp${D_P}_mp${M_P}_pp${P_P}_mbs${MICRO_BATCH_SIZE}_gbs${GLOABAL_BATCH_SIZE}_l${NUM_LAYERS}_hsz${HIDDEN_SIZE}_ahs${NUM_ATTENTION_HEADS}_pretrain_${NODE_RANK}.log

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
       --tensor-model-parallel-size $M_P \
       --pipeline-model-parallel-size $P_P \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_ATTENTION_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOABAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_ITERS \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file ${PROJECT_ROOT}/gpt2-vocab.json \
       --merge-file ${PROJECT_ROOT}/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 100000 \
       --eval-interval 10000 \
       --eval-iters 10 \
       --fp16 2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"
