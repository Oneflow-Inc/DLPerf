WORKSPACE="/workspace/bert"
DATA_DIR="data"
BATCH_SIZE=${1:-32}
DTYPE=${2:-'fp32'}
USE_XLA=${3:-'fasle'}
NUM_TEST=${4:-5}

GPUS_PER_NODE=8
NODE1=10.11.0.2:$GPUS_PER_NODE
NODE2=10.11.0.3:$GPUS_PER_NODE
NODES=${4:-$NODE1,$NODE2}

i=1
while [ $i -le $NUM_TEST ]
do
  bash ${WORKSPACE}/scripts/multi_node_run_pretraining_adam.sh  ${DATA_DIR}  8 ${BATCH_SIZE}  120   $DTYPE   $USE_XLA    $NODES $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
