#!/bin/bash

suffix=$1
log_root=./log
bsz=16384
plan_generator=/hugectr/tools/plan_generation_no_mpi/plan_generator_no_mpi.py

#declare -a num_nodes_list=(4 2 1 1 1 1)
#declare -a num_gpus_list=(8 8 8 4 2 1)
declare -a num_nodes_list=(4 ) 
declare -a num_gpus_list=(8 )
len=${#num_nodes_list[@]}

for (( i=0; i<$len; i++ ))
do
    num_nodes=${num_nodes_list[$i]}
    num_gpus_per_node=${num_gpus_list[$i]}
    gpu_num=$(( ${bsz} * ${num_gpus_per_node} ))

    echo "${num_nodes} nodes ${gpu_num} devices test, total batch size is:${bsz}"
    test_case=${log_root}/n${num_nodes}g${num_gpus_per_node}-fix_total_bsz-${bsz}-${suffix}
    output_json_file=${test_case}.json
    mem_usage_file=${test_case}.mem
    hugectr_log_file=${test_case}.log
   
    # prepare hugeCTR conf json
    python3 gen_hugectr_conf_json.py \
      --template_json wdl_7x1024.json \
      --output_json $output_json_file \
      --total_batch_size $bsz \
      --num_nodes $num_nodes \
      --gpu_num_per_node ${num_gpus_per_node} \
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
    # ./huge_ctr --train $output_json_file >$hugectr_log_file 
    mpirun -np ${num_nodes} \
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
