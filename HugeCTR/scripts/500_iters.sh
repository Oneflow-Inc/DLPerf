bsz=512
log_root=./log

test_case=${log_root}/n1g1-500iters-$1
output_json_file=${test_case}.json
mem_usage_file=${test_case}.mem
hugectr_log_file=${test_case}.log

# prepare hugeCTR conf json
python3 gen_hugectr_conf_json.py \
  --template_json wdl_2x1024.json \
  --output_json $output_json_file \
  --total_batch_size $bsz \
  --num_nodes 1 \
  --gpu_num_per_node 1 \
  --max_iter 500 \
  --display 1 \
  --eval_interval 1 \
  --eval_batches 20 \
  --deep_slot_type Distributed

# watch device 0 memory usage
python3 gpu_memory_usage.py 1>$mem_usage_file 2>&1 </dev/null &

# start hugectr
./huge_ctr --train $output_json_file >$hugectr_log_file 
#./huge_ctr_mn --train $output_json_file >$hugectr_log_file 

