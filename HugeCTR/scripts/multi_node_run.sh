bsz=131072
log_root=/workspace/log

num_nodes=4
gpu_num_per_node=8
NUM_GPU=$(( ${gpu_num_per_node} * ${num_nodes} ))

test_case=${log_root}/n${num_nodes}g${NUM_GPU}-bsz${bsz}
output_json_file=${test_case}.json
mem_usage_file=${test_case}.mem
hugectr_log_file=${test_case}.log

python3 gen_hugectr_conf_json.py \
  --template_json wdl_2x1024.json \
  --output_json $output_json_file \
  --total_batch_size $bsz \
  --num_nodes $num_nodes \
  --gpu_num_per_node $gpu_num_per_node \
  --max_iter 1100 \
  --display 100 \
  --deep_slot_type Distributed

# watch device 0 memory usage
#python3 gpu_memory_usage.py 1>$mem_usage_file 2>&1 </dev/null &

# start hugectr
node1=10.11.0.2
node2=10.11.0.3
node3=10.11.0.4
node4=10.11.0.5
#mpirun -np ${NUM_GPU} \
#  -H ${node2}:8,${node3}:8 \
#  -H ${node1}:1,${node2}:1,${node3}:1,${node4}:1 \
#  -H ${node2}:1,${node3}:1 \
mpirun -np ${num_nodes} \
  -H ${node1}:1,${node2}:1,${node3}:1,${node4}:1 \
  --allow-run-as-root \
  -mca plm_rsh_args "-p2277 -q -o StrictHostKeyChecking=no" \
  -mca btl tcp,self \
  -mca btl_tcp_if_include ib0 \
  -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
  -x NCCL_SOCKET_IFNAME=ib0 \
  --bind-to none \
  --tag-output \
  ./huge_ctr --train $output_json_file 
  #/hugectr/build/bin/huge_ctr --train $output_json_file 
  #-x NCCL_DEBUG=INFO \
  #-x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO \
  #-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
