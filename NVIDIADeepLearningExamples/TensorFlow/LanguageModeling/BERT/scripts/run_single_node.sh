WORKSPACE="/workspace/bert"
DATA_DIR="data"
BATCH_SIZE=${1:-32}
DTYPE=${2:-'fp32'}
USE_XLA=${3:-'false'}
NUM_TEST=${4:-5}

i=1
while [ $i -le $NUM_TEST ]
do
  bash ${WORKSPACE}/scripts/run_pretraining_adam.sh  ${DATA_DIR}  1 ${BATCH_SIZE}  120   $DTYPE   $USE_XLA   $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le $NUM_TEST  ]
do
  bash ${WORKSPACE}/scripts/run_pretraining_adam.sh  ${DATA_DIR}  4 ${BATCH_SIZE} 120   $DTYPE   $USE_XLA    $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le $NUM_TEST  ]
do
  bash ${WORKSPACE}/scripts/run_pretraining_adam.sh  ${DATA_DIR}  8  ${BATCH_SIZE}  120    $DTYPE   $USE_XLA   $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
