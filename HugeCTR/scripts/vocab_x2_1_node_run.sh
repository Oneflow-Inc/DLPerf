#!/bin/bash
suffix=$1

log_root=./log
bsz=16384
plan_generator=/hugectr/tools/plan_generation_no_mpi/plan_generator_no_mpi.py

for vocab_size in 3200000 6400000 12800000 25600000 51200000 
do
    echo "1 node 8 device test, vocab size is:${vocab_size}"
    test_case=${log_root}/n1g8-vocab_x2-${vocab_size}-${suffix}
    output_json_file=${test_case}.json
    mem_usage_file=${test_case}.mem
    hugectr_log_file=${test_case}.log
   
    # prepare hugeCTR conf json
    python3 gen_hugectr_conf_json.py \
      --template_json wdl_7x1024.json \
      --output_json $output_json_file \
      --total_batch_size $bsz \
      --num_nodes 1 \
      --gpu_num_per_node 8 \
      --max_iter 1100 \
      --display 100 \
      --wide_vocab_size $vocab_size \
      --deep_vocab_size $vocab_size \
      --deep_slot_type Localized
    
    # generate plan file all2all_plan_bi_1.json
    #python3 $plan_generator $output_json_file
    

    # watch device 0 memory usage
    python3 gpu_memory_usage.py 1>$mem_usage_file 2>&1 </dev/null &
    
    # start hugectr
    /hugectr/build/bin/huge_ctr --train $output_json_file >$hugectr_log_file 

    sleep 3
done
