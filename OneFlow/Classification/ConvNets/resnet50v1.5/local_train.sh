#!/bin/bash

set +x
num_nodes=${1:-1}
num_gpus=${2:-1}
bsz=${3:-32}
use_fp16=${4:-0}
suffix=${5:-0}

NUM_ITERS=120
LOG_FOLDER=/workspace/log
mkdir -p $LOG_FOLDER
SRC_ROOT=/workspace/git-repos/OneFlow-Benchmark/Classification/cnns
DATA_ROOT=/workspace/dataset/ImageNet/ofrecord

export PYTHONUNBUFFERED=1


NUM_EXAMPLES=$(($num_nodes * $num_gpus * $bsz * $NUM_ITERS))
test_case=n${num_nodes}_g${num_gpus}_b${bsz}
if [[ $use_fp16 -eq 1 ]]; then
    test_case+="_amp"
fi
cmd="/opt/conda/bin/python3 ${SRC_ROOT}/of_cnn_train_val.py "

if [[ $use_fp16 -eq 1 ]]; then
    cmd+="--use_fp16 "
    cmd+="--channel_last=True "
fi

# if [[ -d $DATA_ROOT ]]; then
#     cmd+="--train_data_dir=$DATA_ROOT/train "
#     cmd+="--train_data_part_num=256 "
# fi
cmd+="--num_nodes=${num_nodes} "
cmd+="--gpu_num_per_node=${num_gpus} "
cmd+="--batch_size_per_device=${bsz} "
cmd+="--node_ips=10.244.111.4,10.244.1.14 "
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
logfile=${test_case}_${suffix}_${HOSTNAME}.log
cmd+="--model=resnet50"
#cmd+=" > $LOG_FOLDER/${logfile}"
#cmd+="--model=resnet50 2>&1 | tee $LOG_FOLDER/${logfile}"
$cmd 2>&1 | tee $LOG_FOLDER/${logfile}

set -x

