rm core.*

test_name=$1
DEVICE_NUM_PER_NODE=$2
BATHSIZE=$3
emb_size=$4

MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0
# DATA_DIR=/dataset/wdl_ofrecord/ofrecord
dataset_format=ofrecord
DATA_DIR=/dataset/f9f659c5/wdl_ofrecord
EMBD_SIZE=33762577 # 33762578

# test: 3274330
# val: 3274328
# train: 39291958
eval_batch_size=327432
eval_batchs=$(( 3274330 / eval_batch_size ))

test_case=${test_name}_n1g${DEVICE_NUM_PER_NODE}_BATHSIZE${BATHSIZE}_embsize${emb_size}
log_file=${test_case}.log
mem_file=${test_case}.mem
# export CUDA_VISIBLE_DEVICES=1
export ONEFLOW_DEBUG_MODE=True

python3 gpu_memory_usage.py 1>log/$mem_file 2>&1 </dev/null &

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    train.py \
    --interaction_type dot \
    --dataset_format $dataset_format \
    --embedding_type Embedding \
    --bottom_mlp 512,256,$emb_size \
    --top_mlp 1024,1024,512,256 \
    --embedding_vec_size $emb_size \
    --learning_rate 0.1 \
    --batch_size $BATHSIZE \
    --data_dir $DATA_DIR \
    --loss_print_every_n_iter 100 \
    --eval_interval 1000000 \
    --eval_batchs $eval_batchs \
    --eval_batch_size $eval_batch_size \
    --max_iter 10000 \
    --vocab_size $EMBD_SIZE \
    --data_part_num 256 \
    --data_part_name_suffix_length 5 \
    --execution_mode 'graph' \
    --test_name 'train_graph_conisitent_'$DEVICE_NUM_PER_NODE'gpu' | tee log/${test_case}.log
    # --model_load_dir /tank/model_zoo/dlrm_baseline_params_emb$emb_size \
    # --dataset_format torch \
    # --model_load_dir /tank/xiexuan/dlrm/initial_parameters \
