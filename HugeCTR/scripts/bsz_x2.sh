for bsz in 512 1024 2048 4096 8192 16384 32768 65536 131072 262144; do
    for ngpu in 1 8
    do
      nhidden=2
      test_case=bsz_x2_n1g${ngpu}_b${bsz}_h${nhidden}
      mem_usage_file=${test_case}.mem

      python gpu_memory_usage.py 1> log/$mem_usage_file 2>&1 </dev/null &
      
      python wdl.py \
          --wide_workspace_size_per_gpu_in_mb 9 \
          --deep_workspace_size_per_gpu_in_mb 142 \
          --gpu_num_per_node=$ngpu \
          --hidden_units_num=$nhidden \
          --batch_size=$bsz \
          --max_iter=1200 \
          --eval_interval=10000 | tee log/${test_case}.log
    done
done
