SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BATCH_SIZE=${1:-32}
NODE1='10.11.0.2:9999'     
NODE2='10.11.0.3:9999'
NODE3='10.11.0.4:9999'
NODE4='10.11.0.5:9999'    
CURRENT_NODE=$NODE1
NODES=$NODE1,$NODE2

i=1
while [ $i -le 6 ]
do
  bash $SHELL_FOLDER/multi_node_train.sh "bert_base"    $BATCH_SIZE   0,1,2,3,4,5,6,7  $NODES  $CURRENT_NODE  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done