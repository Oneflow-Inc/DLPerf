#!/bin/bash

set +x
REPEAT_TIMES=1
SHELL_DIR=/workspace/git-repos/DLPerf/OneFlow/Classification/ConvNets/resnet50v1.5

export PYTHONUNBUFFERED=1

#declare -a num_nodes_list=(1 1 2 4)
#declare -a num_gpus_list=(1 8 8 8)
#declare -a num_nodes_list=(1 1 2 )
#declare -a num_gpus_list=(1 2 2 )
#declare -a num_nodes_list=(1 1 )
#declare -a num_gpus_list=(1 8 )
declare -a num_nodes_list=(2 )
declare -a num_gpus_list=(8 )
len=${#num_nodes_list[@]}

#readarray host_arr <hosts
host_arr=($(awk -F= '{print $1}' hosts))
function join { local IFS="$1"; shift; echo "$*"; }

for amp in 0 1
do
    if [[ $amp -eq 1 ]]; then
        declare -a bsz_list=(64 128 256)
    else
        declare -a bsz_list=(32 64 128)
    fi

    for bsz in ${bsz_list[@]}
    do
        for (( i=0; i<$len; i++ ))
        do
            num_nodes=${num_nodes_list[$i]}
            num_gpus=${num_gpus_list[$i]}

            hosts=$(join , ${host_arr[@]::${num_nodes}})
            for (( j=0; j<$REPEAT_TIMES; j++ ))
            do
                cmd="ansible all -i ${hosts}, "
                cmd+="-m shell "
                cmd+="-a \""
                cmd+="chdir=${SHELL_DIR} "
                cmd+="bash local_train.sh ${num_nodes} ${num_gpus} ${bsz} ${amp} ${j} ${hosts} `which python3`"
                cmd+=\"
                echo $cmd
                eval $cmd
                #sleep 130s
            done
        done
    done
done

set -x

