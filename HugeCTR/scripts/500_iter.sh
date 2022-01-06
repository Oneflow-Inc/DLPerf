bsz=512
  
test_case=500iters-n1g1
mem_usage_file=${test_case}.mem

python gpu_memory_usage.py 1> log/$mem_usage_file 2>&1 </dev/null &

python wdl.py \
      --gpu_num_per_node=1 \
      --batch_size=$bsz \
      --eval_batchs=20 \
      --eval_interval 1 \
      --max_iter 500 \
      --loss_print_every_n_iter 1 \
      --hidden_units_num 2 \
      --hidden_size 1024  | tee log/${test_case}.log
