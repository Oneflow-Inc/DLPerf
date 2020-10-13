WORKSPACE="/workspace/rn50v15_tf"
DATA_DIR="/data"
BATCH_SIZE=${1:-128}
DTYPE=${2:-"fp32"}
NUM_STEP=${3:-120}
NUM_TEST=${4:-5}


i=1
while [ $i -le $NUM_TEST ]
do
  USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE}   ${DATA_DIR}    1   $NUM_STEP   $BATCH_SIZE   $DTYPE   $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le $NUM_TEST ]
do
  USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE}   ${DATA_DIR}    4   $NUM_STEP   $BATCH_SIZE   $DTYPE   $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le $NUM_TEST ]
do
  USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE}   ${DATA_DIR}    8   $NUM_STEP   $BATCH_SIZE   $DTYPE   $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done