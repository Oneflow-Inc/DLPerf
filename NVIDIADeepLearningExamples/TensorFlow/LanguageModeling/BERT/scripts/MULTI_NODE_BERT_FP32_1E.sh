BATCH_SIZE=${1:-32}
WORKSPACE=${2:-"/workspace/bert"}
DATA_DIR=${3:-"data"}
NODE1=10.11.0.2:8
NODE2=10.11.0.3:8
NODE3=10.11.0.4:8
NODE4=10.11.0.5:8
NODES=${4:-$NODE1,$NODE2,$NODE3,$NODE4}


i=1
while [ $i -le 6 ]
do
  bash ${WORKSPACE}/scripts/multi_node_run_pretraining_adam.sh  ${DATA_DIR}  8  ${BATCH_SIZE}  120   20 128  $NODES  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
