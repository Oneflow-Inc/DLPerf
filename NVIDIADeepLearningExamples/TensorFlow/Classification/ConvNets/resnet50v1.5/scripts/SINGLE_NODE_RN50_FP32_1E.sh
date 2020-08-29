WORKSPACE=${1:-"/workspace/rn50v15_tf"}
DATA_DIR=${2:-"/data"}

i=1
while [ $i -le 6 ]
do
  USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE} ${DATA_DIR}  1 120 128 fp32 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 6 ]
do
  USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE} ${DATA_DIR}  4 120 128 fp32 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 6 ]
do
  USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE} ${DATA_DIR}  8 120 128 fp32 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done