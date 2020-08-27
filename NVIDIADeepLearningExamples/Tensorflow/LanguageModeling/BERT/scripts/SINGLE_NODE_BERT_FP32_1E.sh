BATCH_SIZE=${1:-32}
WORKSPACE=${2:-"/workspace/bert"}
DATA_DIR=${3:-"data"}

i=1
while [ $i -le 6 ]
do
  bash ${WORKSPACE}/scripts/run_pretraining_adam.sh  ${DATA_DIR}  1 ${BATCH_SIZE}  120   20 128 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 6 ]
do
  bash ${WORKSPACE}/scripts/run_pretraining_adam.sh  ${DATA_DIR}  2 ${BATCH_SIZE}  120   20 128 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 6 ]
do
  bash ${WORKSPACE}/scripts/run_pretraining_adam.sh  ${DATA_DIR}  4 ${BATCH_SIZE} 120   20 128 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 6 ]
do
  bash ${WORKSPACE}/scripts/run_pretraining_adam.sh  ${DATA_DIR}  8  ${BATCH_SIZE}  120   20 128 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
