declare -a num_nodes_list=(1 1 1 1)
declare -a num_gpus_list=(8 4 2 1)
len=${#num_nodes_list[@]}


for repeat_id in 1 2 3
do
    for (( i=0; i<$len; i++ ))
    do
        num_nodes=${num_nodes_list[$i]}
        num_gpus_per_node=${num_gpus_list[$i]}
        gpu_num=$(( ${num_nodes} * ${num_gpus_per_node} ))
        bsz=$(( 16384 * ${gpu_num} ))
        echo $bsz $num_nodes $num_gpus_per_node $gpu_num
        # case 3: fix total bsz
        bash train_nn_graph.sh $num_nodes $num_gpus_per_node 16384 2322444 7 32 fix_total_bsz $repeat_id
        # case 4: fix bsz per device
        bash train_nn_graph.sh $num_nodes $num_gpus_per_node $bsz 2322444 7 32 fix_bsz_per_device $repeat_id
        sleep 3
    done
done