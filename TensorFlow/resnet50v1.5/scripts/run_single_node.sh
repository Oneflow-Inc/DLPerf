BATCH_SIZE=${1:-128}
DTYPE=${2:-"fp32"}
TEST_NUM=${3:-5}
SHELL_FOLDER=$(dirname $(readlink -f "$0"))


i=1
while [ $i -le $TEST_NUM ]
do
  bash $SHELL_FOLDER/single_node_train.sh  0 ${BATCH_SIZE}  $i   $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le $TEST_NUM ]
do
  bash $SHELL_FOLDER/single_node_train.sh  0,1 ${BATCH_SIZE}  $i    $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le  $TEST_NUM ]
do
  bash $SHELL_FOLDER/single_node_train.sh  0,1,2,3 ${BATCH_SIZE}  $i    $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le $TEST_NUM ]
do
  bash $SHELL_FOLDER/single_node_train.sh   0,1,2,3,4,5,6,7 ${BATCH_SIZE}  $i    $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
