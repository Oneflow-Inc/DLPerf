#!/usr/bin/bash
MODEL=${1:-"r100"}
BZ_PER_DEVICE=${2:-64}
ITER_NUM=${3:-120}
GPUS=${4:-0}
NODE_NUM=${5:-1}
SAMPLE_RATIO=${6:-1.0}
DTYPE=${7:-"fp32"}
TEST_NUM=${8:-1}

a=`expr ${#GPUS} + 1`
gpu_num_per_node=`expr ${a} / 2`
gpu_num=`expr ${gpu_num_per_node} \* ${NODE_NUM}`
total_bz=`expr ${BZ_PER_DEVICE} \* ${gpu_num}`

export CUDA_VISIBLE_DEVICES=${GPUS}
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_ALLGATHER=NCCL
export HOROVOD_GPU_BROADCAST=NCLL
export HOROVOD_CACHE_CAPACITY=0
export MXNET_CPU_WORKER_NTHREADS=3

if [ ${NODE_NUM} -eq 1 ] ; then
    node_ip=localhost:${gpu_num_per_node}
elif [ ${NODE_NUM} -eq 2 ] ; then
    node_ip=${NODE1}:${gpu_num_per_node},${NODE2}:${gpu_num_per_node}
elif [ ${NODE_NUM} -eq 4 ] ; then
    node_ip=${NODE1}:${gpu_num_per_node},${NODE2}:${gpu_num_per_node},${NODE3}:${gpu_num_per_node},${NODE4}:${gpu_num_per_node}
else
    echo "Not a valid node."
fi

if [ "$DTYPE" = "fp16" ] ; then
    sed -i "s/\(config.fp16  = \)\S*/config.fp16  = True/" default.py
else
    sed -i "s/\(config.fp16  = \)\S*/config.fp16  = False/" default.py
fi
sed -i "s/\(config.batch_size = \)\S*/config.batch_size = ${BZ_PER_DEVICE}/" default.py
sed -i "s/\(config.max_update = \)\S*/config.max_update = ${ITER_NUM}/" default.py
sed -i "s/\(config.sample_ratio = \)\S*/config.sample_ratio = ${SAMPLE_RATIO}/" default.py


rm -rf checkpoints
log_folder=./logs_1205_sample-ratio-${SAMPLE_RATIO}/mxnet/partial_fc/bz${BZ_PER_DEVICE}/${NODE_NUM}n${gpu_num_per_node}g
mkdir -p $log_folder
log_file=$log_folder/${MODEL}_b${BZ_PER_DEVICE}_${DTYPE}_$TEST_NUM.log

# use `which python` to get the absolute path of your python interpreter
#

# dataset=glint360k_8GPU
dataset=emore

PYTHON_EXEC=/home/leinao/anaconda3/envs/mxnet/bin/python
FOLDER=$(dirname $(readlink -f "$0"))
horovodrun -np ${gpu_num} -H ${node_ip}  ${PYTHON_EXEC} \
${FOLDER}/train_memory.py \
    --dataset ${dataset}  \
    --loss cosface \
    --network ${MODEL}  2>&1 | tee ${log_file}
