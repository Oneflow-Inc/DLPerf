# !/bin/bash 

WORKSPACE=${1:-"/workspace/rn50"}
DATA_DIR=${2:-"/data"}

num_gpus=${3:-1}
bz_per_device=${4:-128}
TRAIN_STEPS=${5:-120}
TEST_TIMES=${6:-1}

MODEL="resnet50"

total_bz=`expr ${bz_per_device} \* ${num_gpus}`
LR=$(awk -v total_bz="$total_bz" 'BEGIN{print  total_bz / 1000}')
NUM_NODES=1

LOG_FOLDER=ngc/pytorch/1n${num_gpus}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/r50_b${bz_per_device}_fp32_$TEST_TIMES.log

CMD="$WORKSPACE/main.py"
CMD+=" --data-backend dali-cpu"
CMD+=" --raport-file $WORKSPACE/raport.json"
CMD+=" -j8 -p 1 --lr $LR"
CMD+=" --optimizer-batch-size -1"
CMD+=" --warmup 8 --arch $MODEL"
CMD+=" -c fanin --label-smoothing 0.1"
CMD+=" --lr-schedule cosine --mom 0.125"
CMD+=" --wd 3.0517578125e-05"
CMD+=" --workspace ${1:-./} -b ${bz_per_device}"
CMD+=" --epochs 1 --prof $TRAIN_STEPS"
CMD+=" --training-only --no-checkpoints"

CMD=" python $WORKSPACE/multiproc.py --nproc_per_node $num_gpus $CMD  $DATA_DIR"

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "Writting log to ${LOGFILE}"
