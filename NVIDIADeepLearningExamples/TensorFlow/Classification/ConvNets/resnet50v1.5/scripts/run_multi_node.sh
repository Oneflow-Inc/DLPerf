GPUS_PER_NODE=8
NODE1=10.11.0.2:$GPUS_PER_NODE
NODE2=10.11.0.3:$GPUS_PER_NODE
NODE3=10.11.0.4:$GPUS_PER_NODE
NODE4=10.11.0.5:$GPUS_PER_NODE

WORKSPACE="/workspace/rn50v15_tf"
DATA_DIR="/data"
BATCH_SIZE=${1:-128}
DTYPE=${2:-"fp32"}
NUM_STEP=${3:-120}
NODES=${4:-$NODE1,$NODE2}
NUM_TEST=${5:-5}



i=1
while [ $i -le $NUM_TEST ]
do
  USE_DALI=1   bash ${WORKSPACE}/resnet50v1.5/training/multi_node_train.sh ${WORKSPACE} ${DATA_DIR} \
  $GPUS_PER_NODE   $NUM_STEP   $BATCH_SIZE   $DTYPE    $NODE1,$NODE2,$NODE3,$NODE4   $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 30
done