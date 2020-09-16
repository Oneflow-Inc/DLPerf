#!/usr/bin/bash
MODEL=${1:-"resnetv15"}
BZ_PER_DEVICE=${2:-32}
ITER_NUM=${3:-120}
GPUS=${4:-0}
NODE_NUM=${5:-1}
DTYPE=${6:-"fp32"}
TEST_NUM=${7:-1}

a=`expr ${#GPUS} + 1`
gpu_num_per_node=`expr ${a} / 2`
gpu_num=`expr ${gpu_num_per_node} \* ${NODE_NUM}`
total_bz=`expr ${BZ_PER_DEVICE} \* ${gpu_num}`

if  [ "$DTYPE" == "fp16" ] ; then
  PRECISION="float16"
else
  PRECISION="float32"
fi


log_folder=logs/ngc/mxnet/resnet50/bz${BZ_PER_DEVICE}/${NODE_NUM}n${gpu_num_per_node}g
mkdir -p $log_folder
log_file=$log_folder/r50_b${BZ_PER_DEVICE}_${DTYPE}_$TEST_NUM.log


if [ ${NODE_NUM} -eq 1 ] ; then
    node_ip=localhost:${gpu_num_per_node}
elif [ ${NODE_NUM} -eq 2 ] ; then
	node_ip=${NODE1}:${gpu_num_per_node},${NODE2}:${gpu_num_per_node}
elif [ ${NODE_NUM} -eq 4 ] ; then
	node_ip=${NODE1}:${gpu_num_per_node},${NODE2}:${gpu_num_per_node},${NODE3}:${gpu_num_per_node},${NODE4}:${gpu_num_per_node}
else
    echo "Not a valid node."
fi

echo ${node_ip}

CMD=""
case $PRECISION in
    "float32") CMD+="--dtype float32 --input-layout NHWC --fuse-bn-relu 0 --fuse-bn-add-relu 0 ";;
    "float16") CMD+="--dtype float16 --fuse-bn-relu 1 --fuse-bn-add-relu 1 \
                  --input-layout NCHW --conv-layout NHWC --batchnorm-layout NHWC \
                  --pooling-layout NHWC ";;
esac

CMD+="--arch resnetv15 \
--num-layers 50 \
--num-classes 1000 \
--mode train \
--data-train ${DATA_DIR}/train.rec \
--data-train-idx ${DATA_DIR}/train.idx \
--gpus ${GPUS} \
--batch-size ${total_bz} \
--image-shape 3,224,224 \
--lr 0.256 \
--lr-schedule cosine \
--optimizer sgd \
--mom 0.875 \
--wd 3.0517578125e-05 \
--label-smoothing 0.1 \
--kv-store horovod \
--data-backend dali-gpu \
--benchmark-iters ${ITER_NUM} \
--no-metrics \
--disp-batches 1 \
--save-frequency 0 \
--num-epochs 1"


echo "begin time: "; date;
# horovodrun -np ${gpu_num} \
# -H ${node_ip} -p ${PORT} \
# --start-timeout 600 \
# python3  train.py ${CMD} 2>&1 | tee ${log_file}


mpirun --allow-run-as-root -oversubscribe -np ${gpu_num} -H ${node_ip} \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    -mca plm_rsh_args "-p ${PORT}  -q -o StrictHostKeyChecking=no" \
    -mca btl_tcp_if_include ib0 \
python3  train.py ${CMD} 2>&1 | tee ${log_file}

echo "Writting to ${log_file}"
echo "end time: "; date;
