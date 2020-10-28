#!/usr/bin/bash
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BZ_PER_DEVICE=${1:-128}
DTYPE=${2:-'fp32'}


export NODE1=10.11.0.5
export NODE2=10.11.0.4
export NODE3=10.11.0.3
export NODE4=10.11.0.2
export PORT=22
echo "BZ_PER_DEVICE >> ${BZ_PER_DEVICE}"


i=1
while [ $i -le 5 ]
do
   bash $SHELL_FOLDER/runner.sh    ${BZ_PER_DEVICE}    0  1   $DTYPE   ${i}
   echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
   let i++
   sleep 20s
done


i=1
while [ $i -le 5 ]
do
   bash $SHELL_FOLDER/runner.sh    ${BZ_PER_DEVICE}    0,1  1   $DTYPE   ${i}
   echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
   let i++
   sleep 20s
done


i=1
while [ $i -le 5 ]
do
   bash $SHELL_FOLDER/runner.sh   ${BZ_PER_DEVICE}    0,1,2,3  1   $DTYPE   ${i}
   echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
   let i++
   sleep 20s
done


i=1
while [ $i -le 5 ]
do
   bash $SHELL_FOLDER/runner.sh   ${BZ_PER_DEVICE}    0,1,2,3,4,5,6,7  1   $DTYPE   ${i}
   echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
   let i++
   sleep 20s
done


i=1
while [ $i -le 5 ]
do
   bash $SHELL_FOLDER/runner.sh   ${BZ_PER_DEVICE}    0,1,2,3,4,5,6,7 2   $DTYPE   ${i}
   echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
   let i++
   sleep 20s
done


i=1
while [ $i -le 5 ]
do
    bash $SHELL_FOLDER/runner.sh   ${BZ_PER_DEVICE} 0,1,2,3,4,5,6,7 4   $DTYPE   ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done