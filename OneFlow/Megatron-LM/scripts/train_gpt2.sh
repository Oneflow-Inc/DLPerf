non_distributed_optimizer=${1:-"off"}
gpus=${2:-0,1,2,3,4,5,6,7}
batch_size_per_device=${3:-4}
dtype=${4:-"fp16"}

a=`expr ${#gpus} + 1`
gpu_num_per_node=`expr ${a} / 2`

export CUDA_VISIBLE_DEVICES=${gpus}
export PYTHONUNBUFFERED=1
export ONEFLOW_DEBUG_MODE=1

# gpt2-small
n_head=12
n_embd=768
n_layer=12
# # gpt2-medium
# n_head=16
# n_embd=1024
# n_layer=24

num_node=4
node_ips="10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"

if [ "$dtype" = "fp16" ] ; then
    use_fp16=True
else
    use_fp16=False
fi

PREFIX=1209-test-19-oneflow-gpt2-small
test_case=${PREFIX}_${dtype}_${num_node}n${gpu_num_per_node}g_bz${batch_size_per_device}_${non_distributed_optimizer}
mem_file=$test_case.mem
log_file=$test_case.log
output_dir=${PREFIX}_output_${non_distributed_optimizer}_distributed_split_${num_node}n${gpu_num_per_node}g_bz${batch_size_per_device}
mkdir -p $output_dir

# nsys is a nvidia analysis tool
# /usr/local/cuda-10.2/bin/nsys  profile
python3 src/train.py \
    --log_dir=${output_dir} \
    --non_distributed_optimizer=${non_distributed_optimizer}  \
    --num_nodes=${num_node} \
    --node_ips=${node_ips}  \
    --use_fp16=${use_fp16}  \
    --dataset=/datasets/wiki/enwiki/AA \
    --batch_size_per_device=${batch_size_per_device}  \
    --gpu_num_per_node=$gpu_num_per_node \
    --seq_len=1024 \
    --iter_num=1000 \
    --optimizer=adamw \
    --embedding_dropout=0.1 \
    --output_dropout=0.1 \
    --attention_dropout=0.1 \
    --n_vocab=50304 \
    --n_ctx=1024  \
    --n_embd=$n_embd \
    --n_head=$n_head \
    --n_layer=${n_layer}   2>&1 | tee ${log_file}
