BATCH_SIZE=${1:-128}
NUM_TESTING=${2:-5}
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
NODE1='10.11.0.2:11111'     
NODE2='10.11.0.3:11111'
NODE3='10.11.0.4:11111'
NODE4='10.11.0.5:11111'    
nodes=$NODE1,$NODE2,$NODE3,$NODE4

i=1
while [ $i -le $NUM_TESTING ]
do
  bash $SHELL_FOLDER/multi_node_train.sh   0,1,2,3,4,5,6,7  ${BATCH_SIZE}   $nodes   $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 30
done