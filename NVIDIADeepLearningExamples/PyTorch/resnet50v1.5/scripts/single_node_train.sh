# !/bin/bash 

WORKSPACE=${1:-"/workspace/rn50"}
DATA_DIR=${2:-"/data"}

num_nodes=${3:-1}
num_gpus=${4:-1}
master_node=${5:-'127.0.0.1'}
master_port=${6:-29500}
bz_per_device=${7:-128}
TRAIN_STEPS=${8:-120}
TEST_TIMES=${9:-1}

MODEL="resnet50"

total_bz=`expr ${bz_per_device} \* ${num_gpus}`
LR=$(awk -v total_bz="$total_bz" 'BEGIN{print  total_bz / 1000}')
NUM_NODES=1

LOG_FOLDER=ngc/pytorch/${num_nodes}n${num_gpus}g
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

CMD=" python $WORKSPACE/multiproc.py  --nnodes ${num_nodes} --node_rank 0 --nproc_per_node ${num_gpus} --master_addr ${master_node} --master_port=${master_port} $CMD $DATA_DIR"

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
