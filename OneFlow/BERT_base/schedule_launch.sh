#!/bin/bash
#
REPEAT_TIMES=7

declare -a num_nodes_list=(1 1 2 4)
declare -a num_gpus_list=(1 8 8 8)
len=${#num_nodes_list[@]}
for bsz in 96 64 32 24 
do
    for (( i=0; i<$len; i++ )) 
    do 
        num_nodes=${num_nodes_list[$i]}  
        num_gpus=${num_gpus_list[$i]}  
        
        # do 7 times for each case
        for (( j=0; j<$REPEAT_TIMES; j++ )) 
        do 
            echo $num_nodes $num_gpus $j $bsz
            ./launch_all.sh $num_nodes $num_gpus $bsz 
            ./cp_logs.sh $num_nodes $num_gpus $bsz $j
        done
    done
done
