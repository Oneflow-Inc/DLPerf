#!/bin/bash
#

##############################################
#0 prepare the host list for training
#comment unused hosts with `#`
#or use first arg to limit the hosts number
#e.g.: `./train.sh 4` will use first 4 hosts.
#declare -a host_list=(
#                  #"10.11.1.1"
#                  "10.11.1.2"
#                  "10.11.1.3"
#                  "10.11.1.4"
#                  "10.11.1.5"
#                  )
declare -a host_list=(ln1 ln2 ln3 ln4)

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
echo "plan to pkill python3 on hosts:${hosts[@]}"

##############################################
#2 copy files to each host and start work
for host in "${hosts[@]}"
do
  echo "pkill python3 on ${host}"
  ssh $USER@$host 'pkill python3'
done

