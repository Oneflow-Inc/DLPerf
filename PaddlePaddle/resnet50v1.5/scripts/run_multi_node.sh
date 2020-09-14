MODEL="resnet50"
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BATCH_SIZE=${1:-128}
DTYPE=${2:-"fp32"}
NODE1='10.11.0.2'     
NODE2='10.11.0.3'
NODE3='10.11.0.4'
NODE4='10.11.0.5'    
CURRENT_NODE=$NODE1
NODES=$NODE1,$NODE2,$NODE3,$NODE4

i=1
while [ $i -le 5 ]
do
  bash $SHELL_FOLDER/multi_node_train.sh  $MODEL    $BATCH_SIZE   0,1,2,3,4,5,6,7  $NODES  $CURRENT_NODE  $i  $DTYPE
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done