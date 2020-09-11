NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BSZ_PER_DEVICE=$3 

if [ -n "$4" ]; then
    NODE_IPS=$4
else
    NODE_IPS='10.11.0.2','10.11.0.3','10.11.0.4','10.11.0.5'
fi

BENCH_ROOT=cnns
DATA_ROOT=/path/to/imagenet_ofrecord
DATA_PART_NUM=32

rm -rf ./log
mkdir ./log

NUM_ITERS=120
NUM_EXAMPLES=$(($NUM_NODES * $GPU_NUM_PER_NODE * $BSZ_PER_DEVICE * $NUM_ITERS))
export PYTHONUNBUFFERED=1
python3 ./$BENCH_ROOT/of_cnn_train_val.py \
    --num_examples=$NUM_EXAMPLES \
    --train_data_dir=$DATA_ROOT/train \
    --train_data_part_num=$DATA_PART_NUM \
    --num_nodes=$NUM_NODES \
    --gpu_num_per_node=$GPU_NUM_PER_NODE \
    --learning_rate=0.001 \
    --loss_print_every_n_iter=20 \
    --batch_size_per_device=$BSZ_PER_DEVICE \
    --val_batch_size_per_device=125 \
    --num_epoch=1 \
    --log_dir=./log \
    --node_ips=$NODE_IPS \
    --model="resnet50"
