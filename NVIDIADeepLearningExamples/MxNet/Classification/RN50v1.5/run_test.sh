#!/usr/bin/bash
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BZ_PER_DEVICE=${1:-128}
DTYPE=${2:-"fp32"}
TEST_NUM=${3:-5}

export DATA_DIR=/data/imagenet/train-val-recordio-passthrough

export NODE1=10.11.0.2
export NODE2=10.11.0.3
export NODE3=10.11.0.4
export NODE4=10.11.0.5
export PORT=10001


i=1
while [ $i -le $TEST_NUM ]
do
    bash $SHELL_FOLDER/runner.sh resnetv15 ${BZ_PER_DEVICE} 120 0 1 ${DTYPE}   ${i}
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    let i++
    sleep 20s
done


i=1
while [ $i -le $TEST_NUM ]
do
    bash $SHELL_FOLDER/runner.sh resnetv15 ${BZ_PER_DEVICE} 120 0,1,2,3 1  ${DTYPE}   ${i}
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    let i++
    sleep 20s
done


i=1
while [ $i -le $TEST_NUM ]
do
    bash $SHELL_FOLDER/runner.sh resnetv15 ${BZ_PER_DEVICE} 120 0,1,2,3,4,5,6,7 1 ${DTYPE}  ${i}
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    let i++
    sleep 20s
done


i=1
while [ $i -le $TEST_NUM ]
do
    bash $SHELL_FOLDER/runner.sh resnetv15 ${BZ_PER_DEVICE} 120 0,1,2,3,4,5,6,7 2 ${DTYPE}   ${i}
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    let i++
    sleep 20s
done


i=1
while [ $i -le $TEST_NUM ]
do
    bash $SHELL_FOLDER/runner.sh resnetv15 ${BZ_PER_DEVICE} 120 0,1,2,3,4,5,6,7 4 ${DTYPE}  ${i}
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    let i++
    sleep 20s
done
