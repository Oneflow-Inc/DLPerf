BATCH_SIZE=${1:-32}
NUM_TESTING=${2:-5}
DTYPE=${3:-'fp32'}
SHELL_FOLDER=$(dirname $(readlink -f "$0"))


i=1
while [ $i -le $NUM_TESTING ]
do
  bash $SHELL_FOLDER/single_node_train.sh  0 ${BATCH_SIZE}     $DTYPE   120  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le $NUM_TESTING ]
do
  bash $SHELL_FOLDER/single_node_train.sh  0,1 ${BATCH_SIZE}   $DTYPE   120  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le  $NUM_TESTING ]
do
  bash $SHELL_FOLDER/single_node_train.sh  0,1,2,3  ${BATCH_SIZE} $DTYPE   120  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le $NUM_TESTING ]
do
  bash $SHELL_FOLDER/single_node_train.sh   0,1,2,3,4,5,6,7 ${BATCH_SIZE}  $DTYPE  120  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done