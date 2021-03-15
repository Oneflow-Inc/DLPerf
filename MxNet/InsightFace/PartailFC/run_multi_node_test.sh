#!/usr/bin/bash
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
MODEL=${1:-r100}
BZ_PER_DEVICE=${2:-64}
SAMPLE_RATIO=${3:-0.1}
DTYPE=${4:-'fp32'}
TEST_NUM=${5:-5}

export NODE1=10.11.0.2
export NODE2=10.11.0.3
export NODE3=10.11.0.4
export NODE4=10.11.0.5


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} 120 0,1,2,3,4,5,6,7  2  $SAMPLE_RATIO  $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} 120 0,1,2,3,4,5,6,7  4  $SAMPLE_RATIO  $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done
