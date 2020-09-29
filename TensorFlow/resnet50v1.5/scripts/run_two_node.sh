BATCH_SIZE=${1:-128}
DTYPE=${2:-"fp32"}
TEST_NUM=${3:-5}
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
NODE1='10.11.0.2:11111'     
NODE2='10.11.0.3:11111'

i=1
while [ $i -le $TEST_NUM ]
do
  bash $SHELL_FOLDER/two_node_train.sh  0,1,2,3,4,5,6,7  ${BATCH_SIZE}   $NODE1,$NODE2   $i  $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 30
done
