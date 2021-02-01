#!/bin/bash

BATCH_SIZE=${1:-128}
DTYPE=${2:-'fp32'}
NUM_TESTING=${3:-5}
SHELL_FOLDER=$(dirname $(readlink -f "$0"))

i=1
while [ $i -le $NUM_TESTING ]
do
  bash $SHELL_FOLDER/scripts/run_standalone_train_gpu.sh  0  ${BATCH_SIZE} ${DTYPE}  200  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le $NUM_TESTING ]
do
  bash $SHELL_FOLDER/scripts/run_distribute_train_gpu.sh  0,1  ${BATCH_SIZE}  ${DTYPE}  200  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le  $NUM_TESTING ]
do
  bash $SHELL_FOLDER/scripts/run_distribute_train_gpu.sh  0,1,2,3  ${BATCH_SIZE}  ${DTYPE}  200  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le $NUM_TESTING ]
do
  bash $SHELL_FOLDER/scripts/run_distribute_train_gpu.sh  0,1,2,3,4,5,6,7  ${BATCH_SIZE}  ${DTYPE}  200  $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

