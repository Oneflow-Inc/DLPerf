#!/bin/sh
MODEL_DIR=../output
rm -rf $MODEL_DIR
GPUS=${1:-"0"}
BATCH_SIZE=${2:-128}
TEST_NUM=${3:-1}
DTYPE=${4:-"fp32"}

a=`expr ${#GPUS} + 1`
num_gpu=`expr ${a} / 2`
echo "Use gpus: $GPUS"
echo "Batch size per device : $BATCH_SIZE"


LOG_FOLDER=../logs/tensorflow/resnet50/bz${BATCH_SIZE}/1n${num_gpu}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/rn50_b${BATCH_SIZE}_${DTYPE}_$TEST_NUM.log

if  [ "$DTYPE" == "fp16" ] ; then
  config_file=configs/examples/resnet/imagenet/gpu_fp16.yaml
else
  config_file=configs/examples/resnet/imagenet/gpu.yaml
fi

export PYTHONPATH=$PYTHONPATH:/home/leinao/tensorflow/models-2.3.0
export CUDA_VISIBLE_DEVICES=${GPUS}
DATA_DIR=/datasets/ImageNet/tfrecord

python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type='resnet' \
  --dataset=imagenet \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=${config_file} \
  --params_override=runtime.num_gpus=$num_gpu  2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"