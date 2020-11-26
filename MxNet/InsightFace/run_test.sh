#!/usr/bin/bash
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
MODEL=${1:-r100}
BZ_PER_DEVICE=${2:-64}
DTYPE=${3:-'fp32'}
TEST_NUM=${4:-5}
echo "BZ_PER_DEVICE >> ${BZ_PER_DEVICE}"


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} 120 0  1   $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} 120 0,1  1   $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} 120 0,1,2,3  1   $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh ${MODEL} ${BZ_PER_DEVICE} 120 0,1,2,3,4,5,6,7  1   $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done