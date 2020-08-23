MODEL="resnet50"
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BATCH_SIZE=128
NODE1='10.11.0.2'     
NODE2='10.11.0.3'
NODE3='10.11.0.4'
NODE4='10.11.0.5'    
CURRENT_NODE=$NODE1
nodes=$NODE1,$NODE2,$NODE3,$NODE4

i=5
while [ $i -le 6 ]
do
  bash $SHELL_FOLDER/multi_node_train.sh  resnet50 0,1,2,3,4,5,6,7  ${BATCH_SIZE}  224 $nodes  $CURRENT_NODE  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 30
done