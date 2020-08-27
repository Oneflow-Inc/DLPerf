
# !/bin/bash 

WORKSPACE=${1:-"/workspace/rn50"}
DATA_DIR=${2:-"/data"}

i=1
while [ $i -le 5 ]
do
  bash ${WORKSPACE}/scripts/single_node_train.sh ${WORKSPACE} ${DATA_DIR} 1 128 120  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash ${WORKSPACE}/scripts/single_node_train.sh ${WORKSPACE} ${DATA_DIR}  4 128 120 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 15 ]
do
  bash ${WORKSPACE}/scripts/single_node_train.sh ${WORKSPACE} ${DATA_DIR}  8 128 120 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
