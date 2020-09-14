MODEL="resnet50"
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BATCH_SIZE=${1:-128}
DTYPE=${2:-"fp32"}

i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh $MODEL  0 ${BATCH_SIZE} 224     $i    $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh $MODEL  0,1 ${BATCH_SIZE} 224    $i    $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh $MODEL 0,1,2,3 ${BATCH_SIZE} 224   $i    $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/single_node_train.sh $MODEL  0,1,2,3,4,5,6,7 ${BATCH_SIZE} 224   $i    $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done