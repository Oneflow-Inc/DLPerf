
# !/bin/bash 

WORKSPACE=${1:-"/workspace/rn50"}
DATA_DIR=${2:-"/data/image"}
bz_per_device=${3:-256}
NUM_NODES=1
MASTER_NODE=127.0.0.1
MASTER_PORT=29500
ITER_TIMES=${4:-150}
PREC=fp16

i=1
while [ $i -le 5 ]
do
  bash ${WORKSPACE}/scripts/single_node_train.sh ${WORKSPACE} ${DATA_DIR} ${NUM_NODES} 1 ${MASTER_NODE} ${MASTER_PORT} ${bz_per_device} ${ITER_TIMES} ${PREC} $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash ${WORKSPACE}/scripts/single_node_train.sh ${WORKSPACE} ${DATA_DIR} ${NUM_NODES} 4 ${MASTER_NODE} ${MASTER_PORT} ${bz_per_device} ${ITER_TIMES} ${PREC} $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 5 ]
do
  bash ${WORKSPACE}/scripts/single_node_train.sh ${WORKSPACE} ${DATA_DIR} ${NUM_NODES} 8 ${MASTER_NODE} ${MASTER_PORT} ${bz_per_device} ${ITER_TIMES} ${PREC} $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
