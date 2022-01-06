NUM_NODES=1
DEVICE_NUM_PER_NODE=1
BATHSIZE=512
EMBD_SIZE=2000000
HIDDEN_UNITS_NUM=2
DEEP_VEC_SIZE=16
PREFIX=500iters
MASTER_ADDR=127.0.0.1
NODE_RANK=0
DATA_DIR=/dataset/f9f659c5/wdl_ofrecord
WDL_MODEL_DIR=/dataset/227246e8/wide_and_deep/train.py

log_root=./log
test_case=${log_root}/500iters-n1g1
oneflow_log_file=${test_case}.log
mem_file=${test_case}.mem


python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    gpu_memory_usage.py 1>$mem_file 2>&1 </dev/null &
python3  $WDL_MODEL_DIR \
    --learning_rate 0.001 \
    --batch_size $BATHSIZE \
    --data_dir $DATA_DIR \
    --loss_print_every_n_iter 1 \
    --eval_interval 0 \
    --deep_dropout_rate 0.5 \
    --max_iter 500 \
    --hidden_size 1024 \
    --wide_vocab_size $EMBD_SIZE \
    --deep_vocab_size $EMBD_SIZE \
    --hidden_units_num $HIDDEN_UNITS_NUM \
    --deep_embedding_vec_size $DEEP_VEC_SIZE \
    --data_part_num 256 \
    --data_part_name_suffix_length 5 \
    --execution_mode 'graph' \
    --test_name 'train_eager_graph_'$DEVICE_NUM_PER_NODE'gpu' > $oneflow_log_file
