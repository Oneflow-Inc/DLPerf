rm -rf core.*


num_nodes=$1
gpu_num_per_device=$2
bsz=$3
vocab_size=$4
hidden_units_num=$5
deep_vec_size=$6
#dataset_type=onerec
#dataset_type=ofrecord
dataset_type=$7
#synthetic

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
#export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

DATA_ROOT=/data/wdl_${dataset_type}
numactl --interleave=all \
python3 wdl_train_eval.py \
  --gpu_num $gpu_num_per_device \
  --num_nodes $num_nodes \
  --node_ips='10.11.0.2','10.11.0.3','10.11.0.4','10.11.0.5' \
  --hidden_units_num $hidden_units_num \
  --dataset_format $dataset_type \
  --train_data_dir $DATA_ROOT/train \
  --eval_data_dir $DATA_ROOT/val \
  --train_data_part_num 256 \
  --train_part_name_suffix_length=5 \
  --eval_data_part_num 256 \
  --eval_part_name_suffix_length=5 \
  --max_iter=1100 \
  --loss_print_every_n_iter=100 \
  --eval_interval=10000 \
  --batch_size=$bsz \
  --deep_embedding_vec_size=$deep_vec_size \
  --wide_vocab_size=$vocab_size \
  --deep_vocab_size=$vocab_size 
  #--train_data_dir $DATA_ROOT/train \
  #--eval_data_dir $DATA_ROOT/val \

