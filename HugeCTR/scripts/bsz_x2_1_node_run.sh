for bsz in 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152; do
  nhidden=2
  ngpu=8
  test_case=n1g${ngpu}_b${bsz}_h${nhidden}
  python wdl.py \
      --gpu_num_per_node=$ngpu \
      --hidden_units_num=$nhidden \
      --batch_size=$bsz \
      --max_iter=1200 \
      --eval_interval=10000 | tee log/${test_case}.log
done
