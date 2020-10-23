#!/usr/bin/bash
BATCH_SIZE=${1:-128}
GPUS=${2:-0,1,2,3,4,5,6,7}
NODE_NUM=${3:-1}
DTYPE=${4:-"fp32"}
TEST_NUM=${5:-1}

a=`expr ${#GPUS} + 1`
gpu_num_per_node=`expr ${a} / 2`
gpu_num=`expr ${gpu_num_per_node} \* ${NODE_NUM}`
total_bz=`expr ${BATCH_SIZE} \* ${gpu_num}`

if [ "$DTYPE" = "fp16" ] ; then
    PRECISION="float16"
else
    PRECISION="float32"
fi


log_folder=../logs/mxnet/resnet50/bz${BATCH_SIZE}/${NODE_NUM}n${gpu_num_per_node}g
mkdir -p $log_folder
log_file=$log_folder/rn50_b${BATCH_SIZE}_${DTYPE}_$TEST_NUM.log

if [ ${NODE_NUM} -eq 1 ] ; then
    node_ip=localhost:${gpu_num_per_node}
elif [ ${NODE_NUM} -eq 2 ] ; then
	node_ip=${NODE1}:${gpu_num_per_node},${NODE2}:${gpu_num_per_node}
elif [ ${NODE_NUM} -eq 4 ] ; then
	node_ip=${NODE1}:${gpu_num_per_node},${NODE2}:${gpu_num_per_node},${NODE3}:${gpu_num_per_node},${NODE4}:${gpu_num_per_node}
else
    echo "Not a valid node."
fi

export CUDA_VISIBLE_DEVICES=$GPUS
DATA_DIR=/datasets/ImageNet/MXNet

mpirun --allow-run-as-root -oversubscribe -np ${gpu_num} -H ${node_ip}  \
     -bind-to none -map-by slot \
     -x LD_LIBRARY_PATH -x PATH \
     -mca pml ob1 -mca btl ^openib \
     -mca plm_rsh_args "-p 22  -q -o StrictHostKeyChecking=no" \
     -mca btl_tcp_if_include ib0   python3 train_horovod.py \
    --mode='hybrid'  \
    --model='resnet50_v1'  \
    --use-rec \
    --rec-train=$DATA_DIR/train.rec  \
    --rec-val=$DATA_DIR/val.rec  \
    --batch-size=${BATCH_SIZE}   \
    --dtype=${PRECISION}  \
    --log-interval=1  \
    --save-frequency=10000 \
    --lr=0.001  \
    --momentum=0.875  \
    --wd=0.000030518  \
    --num-epochs=1  \
    --warmup-epochs=1  2>&1 | tee ${log_file}
