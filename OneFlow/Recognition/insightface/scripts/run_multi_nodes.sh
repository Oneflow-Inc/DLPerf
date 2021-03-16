#!/bin/bash

set -ex


workdir=/workdir

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


port=22

SCRIPTS_PATH=${workdir}/oneflow_face
TEST_SCRIPTS=${SCRIPTS_PATH}/scripts
LOCAL_RUN=${SCRIPTS_PATH}/scripts/train_insightface.sh


##############################################
#0 prepare the host list for training
#comment unused hosts with `#`
#or use first arg to limit the hosts number
declare -a host_list=(
                  "10.11.0.2"
                  "10.11.0.3"
                  "10.11.0.4"
                  "10.11.0.5"
                  )

if [ -n "$1" ]
then
  host_num=$1
else
  host_num=${#host_list[@]}
fi


if [ ${host_num} -gt ${#host_list[@]} ]
then
  host_num=${#host_list[@]}
fi

hosts=("${host_list[@]:0:${host_num}}")
echo "Working on hosts:${hosts[@]}"


if [ ${host_num} == 2 ]; then
  sed -i "s/node_ips = \[.*\]/node_ips = \[\"10.11.0.2\", \"10.11.0.3\"\]/g" $SCRIPTS_PATH/sample_config.py
elif [ ${host_num} == 4 ]; then
  sed -i "s/node_ips = \[.*\]/node_ips = \[\"10.11.0.2\", \"10.11.0.3\", \"10.11.0.4\", \"10.11.0.5\"\]/g" $SCRIPTS_PATH/sample_config.py
else
  echo "Please modify parameters in oneflow_face/sample_config.py, run_multi_nodes.sh manually! "
fi


test_case=${host_num}n${gpu_num_per_node}g_b${bz_per_device}_${network}_${dataset}_${loss}
log_file=${test_case}.log

logs_folder=logs
mkdir -p $logs_folder

echo log file: ${log_file}
##############################################
#1 prepare oneflow_temp folder on each host
for host in "${hosts[@]}"
do
  ssh -p ${port} $host "mkdir -p ~/oneflow_temp"
done

##############################################
#2 copy files to each host and start work
for host in "${hosts[@]:1}"
do
  echo "start training on ${host}"
  ssh -p ${port} $host "rm -rf ~/oneflow_temp/*"
  scp -P ${port} -r $SCRIPTS_PATH $LOCAL_RUN $host:~/oneflow_temp

  ssh -p ${port} $host "cd ~/oneflow_temp; nohup bash train_insightface.sh ~/oneflow_temp/oneflow_face ${network} ${dataset} ${loss} ${num_nodes} $bz_per_device $train_unit $train_iter ${gpu_num_per_node} $precision $model_parallel $partial_fc $test_times $sample_ratio $num_classes 1>${log_file} 2>&1 </dev/null &"
done

#3 copy files to master host and start work
host=${hosts[0]}
echo "start training on ${host}"
ssh -p ${port} $host "rm -rf ~/oneflow_temp/*"
scp -P ${port} -r $SCRIPTS_PATH $LOCAL_RUN $host:~/oneflow_temp
ssh -p ${port} $host "cd ~/oneflow_temp; bash train_insightface.sh ~/oneflow_temp/oneflow_face ${network} ${dataset} ${loss} ${num_nodes} $bz_per_device $train_unit $train_iter ${gpu_num_per_node} $precision $model_parallel $partial_fc $test_times $sample_ratio $num_classes 1>${log_file}"

echo "done"

cp ~/oneflow_temp/${log_file} $logs_folder/${log_file}
sleep 3

