declare -a num_nodes_list=(1 1 1 1)
declare -a num_gpus_list=(1 2 4 8)
len=${#num_nodes_list[@]}

for (( i=0; i<$len; i++ ))
do
    num_nodes=${num_nodes_list[$i]}
    num_gpus_per_node=${num_gpus_list[$i]}
    gpu_num=$(( ${num_nodes} * ${num_gpus_per_node} ))
    bsz=$(( 16384 * ${gpu_num} ))
    echo $bsz $num_nodes $num_gpus_per_node $gpu_num
    test_case=fix_bsz_per_device_n1g${num_gpus_per_node}_b${bsz}_h7
    mem_usage_file=${test_case}.mem

    python gpu_memory_usage.py 1> log/$mem_usage_file 2>&1 </dev/null &

    python wdl.py \
        --wide_workspace_size_per_gpu_in_mb 9 \
        --deep_workspace_size_per_gpu_in_mb 284 \
        --gpu_num_per_node ${num_gpus_per_node} \
        --batch_size $bsz \
        --eval_batchs 20 \
        --eval_interval 10000 \
        --max_iter 1200 \
        --loss_print_every_n_iter 100 \
        --deep_embedding_vec_size 32 \
        --hidden_units_num 7 \
        --hidden_size 1024  | tee log/${test_case}.log
done

