
log_root=./log
for bsz in 16384 32768 65536 131072 262144 524288 1048576 
do
    echo "4 node 32 device test, total batch size is:${bsz}"
    test_case=${log_root}/n4g8-bsz${bsz}-$1
    output_json_file=${test_case}.json
    mem_usage_file=${test_case}.mem
    hugectr_log_file=${test_case}.log
    
    # prepare hugeCTR conf json
    python3 gen_hugectr_conf_json.py \
      --template_json wdl_2x1024.json \
      --output_json $output_json_file \
      --total_batch_size $bsz \
      --num_nodes 4 \
      --gpu_num_per_node 8 \
      --max_iter 1100 \
      --display 100 \
      --deep_slot_type Distributed
    
    # watch device 0 memory usage
    python3 gpu_memory_usage.py 1>$mem_usage_file 2>&1 </dev/null &
    
    # start hugectr
    mpirun -np 4 \
      --hostfile hostfile \
      --allow-run-as-root \
      -mca plm_rsh_args "-p2277 -q -o StrictHostKeyChecking=no" \
      -mca btl tcp,self \
      -mca btl_tcp_if_include ib0 \
      -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
      -x NCCL_SOCKET_IFNAME=ib0 \
      --bind-to none \
      --tag-output \
      /hugectr/build/bin/huge_ctr --train $output_json_file >$hugectr_log_file

    sleep 3
done
