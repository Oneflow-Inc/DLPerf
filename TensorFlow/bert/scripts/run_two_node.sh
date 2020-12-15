BATCH_SIZE=${1:-32}
NUM_TESTING=${2:-5}
DTYPE=${3:-'fp32'}
USE_XLA=${4:-'false'}
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
export PYTHONPATH=$PYTHONPATH:/home/leinao/tensorflow/models-2.3.0
NODE1='10.11.0.2:11111'
NODE2='10.11.0.3:11111'
NODE3='10.11.0.4:11111'
NODE4='10.11.0.5:11111'
nodes=$NODE1,$NODE2
task_index=${5:-0}

i=1
while [ $i -le $NUM_TESTING ]
do
  bash $SHELL_FOLDER/single_node_train.sh  0,1,2,3,4,5,6,7  ${BATCH_SIZE}  $DTYPE  120  $USE_XLA  $i  $nodes  $task_index
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
