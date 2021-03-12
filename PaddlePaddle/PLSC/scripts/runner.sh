model=${1:-"r50"}
batch_size_per_device=${2:-128}
gpus=${3:-0}
node_num=${4:-1}
dtype=${5:-"fp32"}
current_node=${6:-$NODE1}
test_num=${7:-1}
a=`expr ${#gpus} + 1`
gpu_num_per_node=`expr ${a} / 2`


if [ ${node_num} -eq 1 ] ; then
    node_ips=${current_node}
elif [ ${node_num} -eq 2 ] ; then
	node_ips=${NODE1},${NODE2}
elif [ ${node_num} -eq 4 ] ; then
	node_ips=${NODE1},${NODE2},${NODE3},${NODE4}
else
    echo "Not a valid node."
fi


log_dir=./logs/paddle-plsc/arcface/bz${batch_size_per_device}/${node_num}n${gpu_num_per_node}g
mkdir -p $log_dir
log_file=$log_dir/${model}_b${batch_size_per_device}_${dtype}_${test_num}.log


sed -i "s/\(ins.set_train_batch_size=\)\S*/ins.set_train_batch_size=${batch_size_per_device}/" train.py

if [ ${gpu_num_per_node} -eq 1 ] ; then
    sed -i "s/\(LOSS_TYPE = \)\S*/LOSS_TYPE = 'arcface'/" train.py
    python3 train.py  2>&1 | tee $log_file
else
    sed -i "s/\(LOSS_TYPE = \)\S*/LOSS_TYPE = 'dist_arcface'/" train.py
    python3 -m paddle.distributed.launch   \
        --cluster_node_ips="${node_ips}"     \
        --node_ip="${current_node}"    \
        --selected_gpus=${gpus}   train.py  2>&1 | tee $log_file
fi


