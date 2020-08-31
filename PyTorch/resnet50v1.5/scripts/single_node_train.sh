# !/bin/bash

WORKSPACE=${1:-"/examples/imagenet"}
DATA_DIR=${2:-"/data"}
MODEL="resnet50"
NODE1=127.0.0.1:11222
master_node=${3:-$NODE1}

gpus=${4:-0}
bz_per_device=${5:-128}
NUM_NODES=${6:-1}
TEST_TIMES=${7:-1}

a=`expr ${#gpus} + 1`
NUM_GPUS=`expr ${a} / 2`
total_bz=`expr ${bz_per_device} \* ${NUM_GPUS}`
LR=$(awk -v total_bz="$total_bz" 'BEGIN{print  total_bz / 1000}')

export CUDA_VISIBLE_DEVICES=${gpus}
LOG_FOLDER=pytorch/${NUM_NODES}n${NUM_GPUS}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/r50_b${bz_per_device}_fp32_$TEST_TIMES.log

CMD="$WORKSPACE/main.py"
CMD+=" --arch $MODEL"
CMD+=" --epochs 1"
CMD+=" --batch-size $total_bz"
CMD+=" --lr $LR --workers 8"
CMD+=" --momentum 0.125"
CMD+=" --print-freq 1"
CMD+=" --multiprocessing-distributed"
CMD+=" --dist-backend  nccl"
CMD+=" --dist-url  tcp://${master_node}"
CMD+=" --world-size ${NUM_NODES}"
CMD+=" --rank 0"

CMD=" python $CMD $DATA_DIR "

if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

echo "Writting log to ${LOGFILE}"
