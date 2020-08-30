MODEL="bert-base"
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BATCH_SIZE=32

i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh ${BATCH_SIZE} 1 120 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished ${MODEL} Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh ${BATCH_SIZE} 2 120 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished ${MODEL} Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh ${BATCH_SIZE} 4 120 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished ${MODEL} Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh ${BATCH_SIZE} 8 120 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished ${MODEL} Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
