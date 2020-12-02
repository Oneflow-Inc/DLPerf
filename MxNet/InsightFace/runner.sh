#!/usr/bin/bash
MODEL=${1:-"r100"}
BZ_PER_DEVICE=${2:-64}
ITER_NUM=${3:-120}
GPUS=${4:-0}
NODE_NUM=${5:-1}
DTYPE=${6:-"fp32"}
TEST_NUM=${7:-1}

a=`expr ${#GPUS} + 1`
gpu_num_per_node=`expr ${a} / 2`
gpu_num=`expr ${gpu_num_per_node} \* ${NODE_NUM}`
total_bz=`expr ${BZ_PER_DEVICE} \* ${gpu_num}`

if [ "$DTYPE" = "fp16" ] ; then
    PRECISION="float16"
else
    PRECISION="float32"
fi

case $MODEL in
    "r100") LOSS=arcface ;;
    "y1") LOSS=arcface ;;
    ## "y1") LOSS=softmax ;;
esac


log_folder=logs-${MODEL}/insightface/arcface/bz${BZ_PER_DEVICE}/${NODE_NUM}n${gpu_num_per_node}g
mkdir -p $log_folder
log_file=$log_folder/${MODEL}_b${BZ_PER_DEVICE}_${DTYPE}_$TEST_NUM.log

if [ ${NODE_NUM} -eq 1 ] ; then
    node_ip=localhost:${gpu_num_per_node}
else
    echo "Not a valid node."
fi

export CUDA_VISIBLE_DEVICES=${GPUS}
sed -i "s/\(default.per_batch_size = \)\S*/\default.per_batch_size = ${BZ_PER_DEVICE}/" config.py



echo "Begin time: "; date;
# train.py or train_parall.py
python -u train_parall.py \
    --network ${MODEL}  \
    --loss ${LOSS} \
    --dataset emore 2>&1 | tee ${log_file}

echo "Writting log to $log_file"
echo "End time: "; date;
