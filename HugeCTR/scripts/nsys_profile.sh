for bsz in 32 16384; do
  for nhidden in 2 7; do
    for ngpu in 1 4; do
      test_case=n1g${ngpu}_b${bsz}_h${nhidden}
      nsys profile --stats=true -o log/${test_case} \
        python wdl.py \
          --gpu_num_per_node=$ngpu \
          --hidden_units_num=$nhidden \
          --batch_size=$bsz \
	        --max_iter=30 \
          --eval_interval=10000 | tee log/${test_case}.log
    done
  done
done
