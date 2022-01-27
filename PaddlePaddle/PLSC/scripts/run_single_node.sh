#!/usr/bin/bash
export NODE1=10.11.0.2
export NODE2=10.11.0.3
export NODE3=10.11.0.4
export NODE4=10.11.0.5
shell_folder=$(dirname $(readlink -f "$0"))
model=${1:-r50}
batch_size_per_device=${2:-128}
dtype=${3:-'fp32'}
current_node=${4:-$NODE1}
test_num=${5:-5}


i=1
while [ $i -le ${test_num} ]
do
    bash $shell_folder/runner.sh ${model} ${batch_size_per_device}  0  1    $dtype  $current_node  ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le ${test_num} ]
do
    bash $shell_folder/runner.sh ${model} ${batch_size_per_device}  0,1,2,3  1   $dtype  $current_node  ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le ${test_num} ]
do
    bash $shell_folder/runner.sh ${model} ${batch_size_per_device}  0,1,2,3,4,5,6,7  1  $dtype $current_node  ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done
