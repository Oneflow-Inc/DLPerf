rm -rf core.*

# ./$LOCAL_RUN $num_nodes $gpu_num_per_device $bsz $vocab_size $hidden_units_num $deep_vec_size $prefix $suffix

num_nodes=$1
gpu_num_per_device=$2
bsz=$3
vocab_size=$4
hidden_units_num=$5
deep_vec_size=$6
prefix=$7
suffix=$8
#dataset_type=onerec
#dataset_type=ofrecord
#synthetic

dataset_type=$8

case=${prefix}_n${num_nodes}_g${gpu_num_per_device}_b${bsz}_h${hidden_units_num}_${suffix}

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
#export NCCL_DEBUG=INFO
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64

export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
export ONEFLOW_THREAD_ENABLE_LOCAL_MESSAGE_QUEUE=1
export ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER=1
export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=1
export ONEFLOW_STREAM_REUSE_CUDA_EVENT=1

DATA_ROOT=/dataset/f9f659c5/wdl_${dataset_type}
numactl --interleave=all \
python3 $BENCHMARK_ROOT/wdl_train_eval.py \
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
  --deep_vocab_size=$vocab_size | tee log/$case.log
