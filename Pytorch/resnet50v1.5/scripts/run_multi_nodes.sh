# !/bin/bash 

WORKSPACE=${1:-"/examples/imagenet"}
DATA_DIR=${2:-"/data"}
PORT=11222
NODE1='10.11.0.2:'${PORT}
MASTER_NODE=$NODE1
TEST_TIMES=${3:-1}
echo ${MASTER_NODE}

bash ${WORKSPACE}/scripts/single_node_train.sh ${WORKSPACE} ${DATA_DIR} ${MASTER_NODE} 0,1,2,3,4,5,6,7 128 4 ${TEST_TIMES}
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${TEST_TIMES}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
