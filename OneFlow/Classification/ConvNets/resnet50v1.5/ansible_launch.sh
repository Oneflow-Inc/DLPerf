#!/bin/bash

set +x
REPEAT_TIMES=1
NUM_ITERS=120
LOG_FOLDER=/workspace/log
mkdir -p $LOG_FOLDER
SRC_ROOT=/workspace/git-repos/OneFlow-Benchmark/Classification/cnns
DATA_ROOT=/dataset/ImageNet/ofrecord


model="resnet50"

export PYTHONUNBUFFERED=1
export NCCL_LAUNCH_MODE=PARALLEL

#declare -a num_nodes_list=(1 1 2 4)
#declare -a num_gpus_list=(1 8 8 8)
declare -a num_nodes_list=(1 1 2 )
declare -a num_gpus_list=(1 2 2 )
len=${#num_nodes_list[@]}

for amp in 0 1
do
    if [[ $amp -eq 1 ]]; then
        declare -a bsz_list=(64 128 256)
        test_case="_amp"
    else
        declare -a bsz_list=(32 64 128)    
        test_case=""
    fi

    for bsz in ${bsz_list[@]}
    do
        for (( i=0; i<$len; i++ ))
        do
            num_nodes=${num_nodes_list[$i]}
            num_gpus=${num_gpus_list[$i]}
    
            NUM_EXAMPLES=$(($num_nodes * $num_gpus * $bsz * $NUM_ITERS))
            for (( j=0; j<$REPEAT_TIMES; j++ ))
            do
                test_case=n${num_nodes}_g${num_gpus}_b${bsz}
                if [[ $amp -eq 1 ]]; then
                    test_case+="_amp"
                fi
                cmd="python3 ${SRC_ROOT}/of_cnn_train_val.py "

                cmd="ansible hosts_$num_nodes -m shell -a "\"$cmd

                if [[ $amp -eq 1 ]]; then
                    cmd+="--use_fp16 "
                    cmd+="--channel_last=True "
                fi

                if [[ -d $DATA_ROOT ]]; then
                    cmd+="--train_data_dir=$DATA_ROOT/train "
                    cmd+="--train_data_part_num=256 "
                fi
                cmd+="--num_nodes=${num_nodes} "
                cmd+="--gpu_num_per_node=${num_gpus} "
                cmd+="--batch_size_per_device=${bsz} "
                cmd+="--optimizer=sgd "
                cmd+="--momentum=0.875 "
                cmd+="--label_smoothing=0.1 "
                cmd+="--learning_rate=1.536 "
                cmd+="--loss_print_every_n_iter=20 "
                cmd+="--val_batch_size_per_device=125 "
                cmd+="--pad_output "
                cmd+="--fuse_bn_relu=True "
                cmd+="--fuse_bn_add_relu=True "
                cmd+="--nccl_fusion_threshold_mb=16 "
                cmd+="--nccl_fusion_max_ops=24 "
                cmd+="--gpu_image_decoder=True "
                cmd+="--num_epoch=1 "
                cmd+="--num_examples=$NUM_EXAMPLES "
                logfile=${test_case}_${j}_\$\{HOSTNAME\}.log
                #cmd+="--model=resnet50 2>&1 | tee ${logfile}"
                cmd+="--model=resnet50"
                cmd+=" > $LOG_FOLDER/${logfile}"
                cmd+=\"
                echo $cmd 
            done
        done
    done
done

set -x

