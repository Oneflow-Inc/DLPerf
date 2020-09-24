MODEL="bert-base"
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BATCH_SIZE=96
NUM_NODES=1
MASTER_NODE=127.0.0.1
MASTER_PORT=29500
PREC=fp16
ITER_NUM=150

i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh ${BATCH_SIZE} ${NUM_NODES} ${MASTER_NODE} ${MASTER_PORT} 1 ${ITER_NUM} ${PREC} $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished ${MODEL} Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh ${BATCH_SIZE} ${NUM_NODES} ${MASTER_NODE} ${MASTER_PORT} 4 ${ITER_NUM} ${PREC} $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished ${MODEL} Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh ${BATCH_SIZE} ${NUM_NODES} ${MASTER_NODE} ${MASTER_PORT} 8 ${ITER_NUM} ${PREC} $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished ${MODEL} Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
