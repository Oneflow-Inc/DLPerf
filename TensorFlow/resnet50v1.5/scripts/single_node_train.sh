#!/bin/sh
MODEL_DIR=../output
rm -rf $MODEL_DIR
gpus=${1:-"0,1"}
bz_per_device=${2:-128}
TEST_NUM=${3:-1}

a=`expr ${#gpus} + 1`
NUM_GPU=`expr ${a} / 2`
echo "Use gpus: $gpus"
echo "Batch size : $bz_per_device"


LOG_FOLDER=../tensorflow2/resnet50/1n${NUM_GPU}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/rn50_b${bz_per_device}_fp32_$TEST_NUM.log

# export PYTHONPATH=$PYTHONPATH:$BENCH_ROOT_DIR/tensorflow/models-2.3.0
export PYTHONPATH=$PYTHONPATH:/home/leinao/tensorflow/models-2.3.0
export CUDA_VISIBLE_DEVICES=${gpus}
DATA_DIR=/datasets/ImageNet/tfrecord

python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type='resnet' \
  --dataset=imagenet \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=configs/examples/resnet/imagenet/gpu.yaml \
  --params_override=runtime.num_gpus=$NUM_GPU  2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"