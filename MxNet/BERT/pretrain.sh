#!/usr/bin/bash
MODEL=${1:-"bert_base"}
BZ_PER_DEVICE=${2:-32}
ITER_NUM=${3:-200}
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

log_folder=logs/mxnet/bert/bz${BZ_PER_DEVICE}/${NODE_NUM}n${gpu_num_per_node}g
mkdir -p $log_folder
log_file=$log_folder/bert_b${BZ_PER_DEVICE}_${DTYPE}_$TEST_NUM.log

if [ ${NODE_NUM} -eq 1 ] ; then
    node_ip=localhost:${gpu_num_per_node}
elif [ ${NODE_NUM} -eq 2 ] ; then
	node_ip=${NODE1}:${gpu_num_per_node},${NODE2}:${gpu_num_per_node}
elif [ ${NODE_NUM} -eq 4 ] ; then
	node_ip=${NODE1}:${gpu_num_per_node},${NODE2}:${gpu_num_per_node},${NODE3}:${gpu_num_per_node},${NODE4}:${gpu_num_per_node}
else
    echo "Not a valid node."
fi


ONE_PART_NPY=$(eval ls ${DATA_DIR}/* | tr " " "\n" | awk '{printf "%s,",$1}' | sed s'/.$//')

CMD=""
case $MODEL in
    "bert_base") CMD+="--model bert_12_768_12 ";;
    "bert_large") CMD+="--model bert_24_1024_16 ";;
esac


CMD+="--dtype ${PRECISION} \
--warmup_ratio 1 \
--comm_backend horovod \
--total_batch_size ${total_bz} \
--total_batch_size_eval ${total_bz} \
--accumulate 1 \
--lr 1e-4 \
--max_seq_length 128 \
--max_predictions_per_seq 20 \
--num_steps ${ITER_NUM} \
--log_interval 1 \
--ckpt_interval 1000 \
--no_compute_acc \
--data ${ONE_PART_NPY} "

echo "begin time: "; date;
# horovodrun -np ${gpu_num} -H ${node_ip}   -p ${PORT} \
# --start-timeout 600 \
# python3 ${WORKSPACE}/run_pretraining.py ${CMD} 2>&1 | tee ${log_file}

mpirun -oversubscribe -np ${gpu_num} -H ${node_ip} \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    -mca plm_rsh_args "-p 22 -q -o StrictHostKeyChecking=no" \
    -mca btl_tcp_if_include ib0 \
python3 ${WORKSPACE}/run_pretraining.py ${CMD} 2>&1 | tee ${log_file}


echo "Writting log to $log_file"
echo "end time: "; date;