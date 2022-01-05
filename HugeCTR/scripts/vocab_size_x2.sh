bsz=16384
  

for i in 1 2 4 8 16
do
    vocab_size=$(( 1798398*${i} ))

    for ngpu in 1 8
    do
        test_case=vocab_x2_n1g${ngpu}_vsz${i}_h7
        mem_usage_file=${test_case}.mem
        wide_workspace_size_per_gpu_in_mb=$(( 8*${i} ))
        deep_workspace_size_per_gpu_in_mb=$(( 114*${i} ))

        python gpu_memory_usage.py 1> log/$mem_usage_file 2>&1 </dev/null &

        python wdl.py \
            --gpu_num_per_node=$ngpu \
            --batch_size=$bsz \
            --eval_batchs=20 \
            --eval_interval 1000 \
            --max_iter 500 \
            --loss_print_every_n_iter 100 \
            --hidden_units_num 7 \
            --wide_workspace_size_per_gpu_in_mb=$wide_workspace_size_per_gpu_in_mb \
            --deep_workspace_size_per_gpu_in_mb=$deep_workspace_size_per_gpu_in_mb \
            --hidden_size 1024  | tee log/${test_case}.log
    done
done
