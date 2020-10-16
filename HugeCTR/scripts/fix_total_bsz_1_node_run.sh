
suffix=$1

log_root=./log
bsz=16384
plan_generator=/hugectr/tools/plan_generation_no_mpi/plan_generator_no_mpi.py

for gpu_num in 1 2 4 8 
do
    echo "1 node ${gpu_num} device test, total batch size is:${bsz}"
    test_case=${log_root}/n1g${gpu_num}-fix_total_bsz-${bsz}-${suffix}
    output_json_file=${test_case}.json
    mem_usage_file=${test_case}.mem
    hugectr_log_file=${test_case}.log
   
    
    # prepare hugeCTR conf json
    python3 gen_hugectr_conf_json.py \
      --template_json wdl_7x1024.json \
      --output_json $output_json_file \
      --total_batch_size $bsz \
      --num_nodes 1 \
      --gpu_num_per_node ${gpu_num} \
      --max_iter 1100 \
      --display 100 \
      --deep_slot_type Localized
    
    #if [ "$gpu_num" -gt 1 ]
    #then
    #    # generate plan file all2all_plan_bi_1.json
    #    python3 $plan_generator $output_json_file
    #fi

    # watch device 0 memory usage
    python3 gpu_memory_usage.py 1>$mem_usage_file 2>&1 </dev/null &
    
    # start hugectr
    /hugectr/build/bin/huge_ctr --train $output_json_file >$hugectr_log_file 

    sleep 3
done
