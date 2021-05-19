#! /bin/bash
# set -ex

export PYTHONUNBUFFERED=1
# export ONEFLOW_DEBUG_MODE=1
# export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1
# export NCCL_DEBUG=INFO

dataset=/data/gpt/gpt_sample_dataset_text_document
seq_length=2048

num_nodes=1
node_ips="10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"
num_gpus_per_node=8

tensor_model_parallel_size=4
pipeline_model_parallel_size=1
world_size=$(($num_gpus_per_node*$num_nodes))
data_parallel_size=$((${world_size}/$tensor_model_parallel_size/$pipeline_model_parallel_size))

num_layers=16
hidden_size=1536
num_heads=16

micro_batch_size=8
global_batch_size=16

train_iters=100
log_interval=10

log_file=pretrain_gpt_${num_nodes}n${num_gpus_per_node}d_dp${data_parallel_size}_mp${tensor_model_parallel_size}_pp${pipeline_model_parallel_size}_mbz${micro_batch_size}_gbz${global_batch_size}_s${seq_length}_l${num_layers}_h${hidden_size}_nh${num_heads}.log

python3 -m oneflow_gpt.training \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_heads} \
    --micro-batch-size ${micro_batch_size} \
    --global-batch-size ${global_batch_size} \
    --tensor-model-parallel-size ${tensor_model_parallel_size} \
    --pipeline-model-parallel-size ${pipeline_model_parallel_size} \
    --num-gpus-per-node ${num_gpus_per_node} \
    --num-nodes ${num_nodes} \
    --node-ips ${node_ips} \
    --train-iters ${train_iters} \
    --dataset ${dataset} \
    --seq-length ${seq_length} \
    --log-interval ${log_interval} \
    --learning-rate 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 320000 \
    --lr-warmup-fraction 0.01 \
    --initial-loss-scale 1048576 \
    --optimizer adamw \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --vocab-size 50257 \
    --split 949,50,1 \
    --load checkpoint \
    --save checkpoint \
    --save-interval 20000 \
    --metric-print-format table \
    --checkpoint-activations \
    --multihead-attention-fusion \
    --fp16 \
    --use-rdma \
    | tee $log_file
