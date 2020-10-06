#!/bin/bash
#

LOCAL_RUN=$1
BENCH_ROOT=$2
NUM_NODES=$3
GPU_NUM_PER_NODE=$4
BSZ=$5

#0 prepare the host list ips for training
declare -a host_list=("10.11.0.2" "10.11.0.3" "10.11.0.4" "10.11.0.5")

if [ $NUM_NODES -gt ${#host_list[@]} ]
then
    echo num_nodes should be less than or equal to length of host_list.
    exit
fi

hosts=("${host_list[@]:0:${NUM_NODES}}")
echo "Working on hosts:${hosts[@]}"

ips=${hosts[0]}
for host in "${hosts[@]:1}" 
do
   ips+=",${host}"
done

#1 prepare oneflow_temp folder on each host
for host in "${hosts[@]}" 
do
  ssh $USER@$host "mkdir -p ~/oneflow_temp"
done

#2 copy files to slave hosts and start work with nohup
for host in "${hosts[@]:1}" 
do
  echo "start training on ${host}"
  ssh $USER@$host 'rm -rf ~/oneflow_temp/*'
  scp -r $BENCH_ROOT ./$LOCAL_RUN $USER@$host:~/oneflow_temp
  ssh $USER@$host "conda activate oneflow; cd ~/oneflow_temp; nohup ./$LOCAL_RUN $NUM_NODES $GPU_NUM_PER_NODE $BSZ $ips 1>oneflow.log 2>&1 </dev/null &"
done

#3 copy files to master host and start work
host=${hosts[0]}
echo "start training on ${host}"
ssh $USER@$host 'rm -rf ~/oneflow_temp/*'
scp -r $BENCH_ROOT ./$LOCAL_RUN $USER@$host:~/oneflow_temp
ssh $USER@$host "conda activate oneflow; cd ~/oneflow_temp; ./$LOCAL_RUN $NUM_NODES $GPU_NUM_PER_NODE $BSZ $ips > oneflow.log"

echo "done"

