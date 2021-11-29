for bsz in 32 16384; do
  for nhidden in 2 7; do
    for ngpu in 1 4; do
      python wdl.py \
          --gpu_num_per_node=$ngpu \
          --hidden_units_num=$nhidden \
          --batch_size=$bsz \
          --max_iter=1200 \
          --eval_interval=10000 | tee n1g$ngpu_b$bsz_h$nhidden.log
    done
  done
done
