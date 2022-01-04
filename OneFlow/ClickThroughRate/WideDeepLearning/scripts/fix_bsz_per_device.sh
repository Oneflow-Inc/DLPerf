declare -a num_nodes_list=(1 1 1 1)
declare -a num_gpus_list=(8 4 2 1)
len=${#num_nodes_list[@]}

NUM_NODES=1
EMBD_SIZE=2000000
HIDDEN_UNITS_NUM=7
DEEP_VEC_SIZE=16
PREFIX=fix_total_bsz
MASTER_ADDR=127.0.0.1
NODE_RANK=0
DATA_DIR=/dataset/f9f659c5/wdl_ofrecord
WDL_MODEL_DIR=/dataset/227246e8/wide_and_deep/train.py

export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
export ONEFLOW_THREAD_ENABLE_LOCAL_MESSAGE_QUEUE=1
export ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER=1
export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=1

for (( i=0; i<$len; i++ ))
do
    num_nodes=${num_nodes_list[$i]}
    num_gpus_per_node=${num_gpus_list[$i]}
    gpu_num=$(( ${num_nodes} * ${num_gpus_per_node} ))
    bsz=$(( 16384 * ${gpu_num} ))
    
    log_root=./log
    test_case=${log_root}/fix_bsz_per_device_n1g${num_gpus_per_node}_b${bsz}_h${HIDDEN_UNITS_NUM}
    oneflow_log_file=${test_case}.log
    mem_file=${test_case}.mem
    batch_size_per_proc=$(( ${bsz}/${num_gpus_per_node} ))
    
    python3 gpu_memory_usage.py 1>$mem_file 2>&1 </dev/null &
    
    python3 -m oneflow.distributed.launch \
            --nproc_per_node $num_gpus_per_node \
            --nnodes $NUM_NODES \
            --node_rank $NODE_RANK \
            --master_addr $MASTER_ADDR \
            $WDL_MODEL_DIR \
            --learning_rate 0.001 \
            --batch_size $bsz \
            --batch_size_per_proc $batch_size_per_proc \
            --data_dir $DATA_DIR \
            --loss_print_every_n_iter 100 \
            --eval_interval 0 \
            --deep_dropout_rate 0.5 \
            --max_iter 1000 \
            --hidden_size 1024 \
            --wide_vocab_size $EMBD_SIZE \
            --deep_vocab_size $EMBD_SIZE \
            --hidden_units_num $HIDDEN_UNITS_NUM \
            --deep_embedding_vec_size $DEEP_VEC_SIZE \
            --data_part_num 256 \
            --data_part_name_suffix_length 5 \
            --execution_mode 'graph' \
            --test_name 'train_eager_graph_'$DEVICE_NUM_PER_NODE'gpu' > $oneflow_log_file
done
