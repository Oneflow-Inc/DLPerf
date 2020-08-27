#!/bin/bash
MODEL_DIR=../output
rm -rf $MODEL_DIR
gpus=${1:-"0,1,2,3,4,5,6,7"}
bz_per_device=${2:-128}
node_ips=${3:-$NODE1,$NODE2,$NODE3,$NODE4}
TEST_NUM=${4:-1}
node_num=$(echo $node_ips | tr ',' '\n' | wc -l)

a=`expr ${#gpus} + 1`
NUM_GPU=`expr ${a} / 2`
echo "Node ip : $node_ips"
echo "Use gpus: $gpus"
echo "Batch size : $bz_per_device"


LOG_FOLDER=../tensorflow2/resnet50/${node_num}n${NUM_GPU}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/rn50_b${bz_per_device}_fp32_$TEST_NUM.log

# export PYTHONPATH=$PYTHONPATH:$BENCH_ROOT_DIR/tensorflow/models-2.3.0
export PYTHONPATH=$PYTHONPATH:/home/leinao/tensorflow/models-2.3.0
export CUDA_VISIBLE_DEVICES=$gpus
DATA_DIR=/datasets/ImageNet/tfrecord # Set up your tfrecord path

python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type='resnet' \
  --dataset=imagenet \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=configs/examples/resnet/imagenet/multi_node_gpu.yaml \
  --params_override='runtime.num_gpus='$NUM_GPU   2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"