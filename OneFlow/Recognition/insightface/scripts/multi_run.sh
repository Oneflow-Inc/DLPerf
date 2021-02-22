#!/bin/bash

set -ex

WORKSPACE=~/oneflow_temp
SCRIPTS_PATH=$WORKSPACE/oneflow_face
host_num=${1:-4}
network=${2:-"r100"}
dataset=${3:-"emore"}
loss=${4:-"arcface"}
num_nodes=${5:-${host_num}}
bz_per_device=${6:-64}
train_unit=${7:-"batch"}
train_iter=${8:-150}
gpu_num_per_node=${9:-8}
precision=${10:-fp32}
model_parallel=${11:-1}
partial_fc=${12:-1}
test_times=${13:-5}
sample_ratio=${14:-0.1}
num_classes=${15:-85744}
use_synthetic_data=${16:-False}

# 2n8g
sed 
sed -i  "s/num_nodes = 1/num_nodes = 2/g" $SCRIPTS_PATH/sample_config.py
sed -i "s/node_ips = \['10.11.0.2'\]/node_ips = \['10.11.0.2', '10.11.0.3'\]/g" $SCRIPTS_PATH/sample_config.py
sed -i "s/\"10.11.0.3\"/\#\"10.11.0.3\"/g" $WORKSPACE/run_multi_nodes.sh
sed -i "s/\"10.11.0.4\"/\#\"10.11.0.4\"/g" $WORKSPACE/run_multi_nodes.sh


i=1
while [ $i -le ${test_times} ]
do
    bash $SCRIPTS_PATH/run_multi_nodes.sh 2 ${network} ${dataset} ${loss} 2 $bz_per_device $train_unit $train_iter ${gpu_num_per_node} $precision $model_parallel $partial_fc $i $sample_ratio $num_classes $use_synthetic_data 
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done

# 4n8g
sed -i  "s/num_nodes = 2/num_nodes = 4/g" $SCRIPTS_PATH/sample_config.py
sed -i "s/node_ips = \['10.11.0.2', '10.11.0.3'\]/node_ips = \['10.11.0.2', '10.11.0.3', '10.11.0.4', '10.11.0.5'\]/g" $SCRIPTS_PATH/sample_config.py
sed -i "s/\#\"10.11.0.3\"/\"10.11.0.3\"/g" $WORKSPACE/run_multi_nodes.sh
sed -i "s/\#\"10.11.0.4\"/\"10.11.0.4\"/g" $WORKSPACE/run_multi_nodes.sh

i=1
while [ $i -le ${test_times} ]
do    bash $SCRIPTS_PATH/run_multi_nodes.sh 4 ${network} ${dataset} ${loss} 4 $bz_per_device $train_unit $train_iter ${gpu_num_per_node} $precision $model_parallel $partial_fc $i $sample_ratio $num_classes $use_synthetic_data         
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done
