#!/bin/bash
#

set -ex

#host_num=$1
gpu_num_per_device=$2
bsz=$3
vocab_size=$4
hidden_units_num=$5
deep_vec_size=$6
prefix=$7
suffix=$8
dataset_type=${9:-onerec}

LOCAL_RUN=local_launch_in_docker.sh
port=12395
WDL_script=/OneFlow-Benchmark/ClickThroughRate/WideDeepLearning/wdl_train_eval.py

##############################################
#0 prepare the host list for training
#comment unused hosts with `#`
#or use first arg to limit the hosts number
declare -a host_list=(
                  "10.11.0.3"
                  "10.11.0.2"
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

test_case=n${host_num}g${gpu_num_per_device}_${prefix}_bsz-${bsz}_vocab-${vocab_size}_${suffix}_${dataset_type}
log_file=${test_case}.log
mem_file=${test_case}.mem
INFO_file=${test_case}.INFO

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
  ssh -p ${port} $host 'rm -rf ~/oneflow_temp/*'
  scp -P ${port} -r $WDL_script ./$LOCAL_RUN $host:~/oneflow_temp
  ssh -p ${port} $host "cd ~/oneflow_temp; nohup ./$LOCAL_RUN $host_num $gpu_num_per_device $bsz $vocab_size $hidden_units_num $deep_vec_size $dataset_type 1>${log_file} 2>&1 </dev/null &"
done

# 2.5 watch device 0 memory usage
python3 gpu_memory_usage.py 1>$logs_folder/$mem_file 2>&1 </dev/null &

#3 copy files to master host and start work
host=${hosts[0]}
echo "start training on ${host}"
ssh -p ${port} $host 'rm -rf ~/oneflow_temp/*'
scp -P ${port} -r $WDL_script ./$LOCAL_RUN $host:~/oneflow_temp
ssh -p ${port} $host "cd ~/oneflow_temp; ./$LOCAL_RUN $host_num $gpu_num_per_device $bsz $vocab_size $hidden_units_num $deep_vec_size $dataset_type 1>${log_file}"

echo "done"

cp ~/oneflow_temp/${log_file} $logs_folder/${log_file}
cp ~/oneflow_temp/log/VS00${host: -1}/oneflow.INFO $logs_folder/${INFO_file}
#python3 extract_time.py --log_file=logs/${log_file}
sleep 3
# kill python3 gpu_memory_usage.py
# pkill python3

