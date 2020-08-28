WORKSPACE=${1:-"/workspace/rn50v15_tf"}
DATA_DIR=${2:-"/data"}

GPUS_PER_NODE=8
NODE1=10.11.0.2:$GPUS_PER_NODE
NODE2=10.11.0.3:$GPUS_PER_NODE
NODE3=10.11.0.4:$GPUS_PER_NODE
NODE4=10.11.0.5:$GPUS_PER_NODE

i=1
while [ $i -le 6 ]
do
  USE_DALI=1   bash ${WORKSPACE}/resnet50v1.5/training/multi_node_train.sh ${WORKSPACE} ${DATA_DIR} \
  $GPUS_PER_NODE 120 128 fp32  $NODE1,$NODE2,$NODE3,$NODE4 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 30
done

# # 4 node 8 gpu
# USE_DALI=1   bash ${WORKSPACE}/resnet50v1.5/training/multi_node_train.sh ${WORKSPACE} ${DATA_DIR} \
#     $GPUS_PER_NODE 120 128 fp32  $NODE1,$NODE2,$NODE3,$NODE4