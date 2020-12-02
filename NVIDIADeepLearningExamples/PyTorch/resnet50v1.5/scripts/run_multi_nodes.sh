# !/bin/bash 

WORKSPACE=${1:-"/workspace/rn50"}
DATA_DIR=${2:-"/data/image"}

NUM_NODES=4
MASTER_NODE=10.11.0.2
MASTER_PORT=22333
bz_per_device=${3:-256}
TRAIN_STEPS=150
PREC=amp

i=1
while [ $i -le 5 ]
do
  bash ${WORKSPACE}/scripts/single_node_train.sh ${WORKSPACE} ${DATA_DIR} ${NUM_NODES} 8 ${MASTER_NODE} ${MASTER_PORT} ${bz_per_device} ${TRAIN_STEPS} ${PREC} $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
