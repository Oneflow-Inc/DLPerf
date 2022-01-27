#!/usr/bin/bash
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
MODEL=${1:-gpt2-small}
BATCH_SIZE_PER_DEVICE=${2:-8}
ZERO_STAGE=${3:-2}
CHECKPOINT_ACTIVATIONS=${4:-"on"}
DTYPE=${5:-'fp16'}
TEST_NUM=${6:-5}



i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh $MODEL $BATCH_SIZE_PER_DEVICE 1 1 $ZERO_STAGE $CHECKPOINT_ACTIVATIONS $DTYPE ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    pkill python3
    sleep 30s
done

i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh $MODEL $BATCH_SIZE_PER_DEVICE 1 4 $ZERO_STAGE $CHECKPOINT_ACTIVATIONS $DTYPE ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    pkill python3
    sleep 30s
done

i=1
while [ $i -le ${TEST_NUM} ]
do
    bash $SHELL_FOLDER/runner.sh $MODEL $BATCH_SIZE_PER_DEVICE 1 8 $ZERO_STAGE $CHECKPOINT_ACTIVATIONS $DTYPE ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    pkill python3
    sleep 30s
done


