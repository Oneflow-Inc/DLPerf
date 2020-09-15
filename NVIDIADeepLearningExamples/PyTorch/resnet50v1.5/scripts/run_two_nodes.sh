
# !/bin/bash 

WORKSPACE=${1:-"/workspace/rn50"}
DATA_DIR=${2:-"/data"}

NUM_NODES=2
MASTER_NODE=10.11.0.2
MASTER_PORT=22333
bz_per_device=128
TRAIN_STEPS=121

i=1
while [ $i -le 5 ]
do
  bash ${WORKSPACE}/scripts/single_node_train.sh ${WORKSPACE} ${DATA_DIR} ${NUM_NODES} 8 ${MASTER_NODE} ${MASTER_PORT} 128 121 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
