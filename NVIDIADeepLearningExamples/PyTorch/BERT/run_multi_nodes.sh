# !/bin/bash 

MODEL="bert-base"
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BATCH_SIZE=48
NUM_NODES=4
MASTER_NODE=10.11.0.2
MASTER_PORT=22334
ITER_NUM=150
PREC=fp32

i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh ${BATCH_SIZE} ${NUM_NODES} ${MASTER_NODE} ${MASTER_PORT} 8 ${ITER_NUM} ${PREC} $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished ${MODEL} Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
