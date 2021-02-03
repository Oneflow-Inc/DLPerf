#!/bin/bash
MODEL_DIR=../output
rm -rf $MODEL_DIR
gpus=${1:-"0"}
BATCH_SIZE=${2:-32}
DTYPE=${3:-'fp32'}
NUM_STEP=${4:-120}
USE_XLA=${5:-'false'}
TEST_NUM=${6:-1}
NODE1='10.11.0.2:11111'
NODE_IPS=${7:-$NODE1}
task_index=${8:-0}

NODE_NUM=$(echo $NODE_IPS | tr ',' '\n' | wc -l)
a=`expr ${#gpus} + 1`
NUM_GPU=`expr ${a} / 2`
total_batch_size=`expr ${BATCH_SIZE} \* $NUM_GPU`
echo "Node ip : $NODE_IPS"
echo "Use gpus: $gpus"
echo "Total batch size : $total_batch_size"


if  [ "$USE_XLA" == "true" ] ; then
  enable_xla='true'
  export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
else
  enable_xla='false'
fi

BERT_BASE_CONFIG_FILE='/datasets/bert/uncased_L-12_H-768_A-12/bert_config.json' 
LOG_FOLDER=./logs/tensorflow/bert/bz${BATCH_SIZE}/${NODE_NUM}n${NUM_GPU}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/bert_b${BATCH_SIZE}_${DTYPE}_${TEST_NUM}.log


export CUDA_VISIBLE_DEVICES=$gpus
CMD="python run_pretraining.py"
CMD+=" --input_files=/datasets/bert/wiki/*.tfrecord"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --train_batch_size=$total_batch_size"
CMD+=" --num_steps_per_epoch=$NUM_STEP"
CMD+=" --num_train_epochs=1"
CMD+=" --warmup_steps=10000"
CMD+=" --use_next_sentence_label=True"
CMD+=" --train_summary_interval=0"
CMD+=" --optimizer_type=adamw"
CMD+=" --num_gpus=$NUM_GPU"
CMD+=" --datasets_num_private_threads=8"
CMD+=" --dtype=$DTYPE"
CMD+=" --enable_xla=$enable_xla"
CMD+=" --model_dir=$MODEL_DIR"
CMD+=" --bert_config_file=${BERT_BASE_CONFIG_FILE}"

if [ $NODE_NUM -gt 1 ] ; then
  CMD+=" --distribution_strategy=multi_worker_mirrored"
  CMD+=" --worker_hosts=$NODE_IPS"
  CMD+=" --task_index=$task_index"
  CMD+=" --all_reduce_alg=nccl"
fi

$CMD 2>&1 | tee $LOGFILE

echo "Writting log to $LOGFILE"
