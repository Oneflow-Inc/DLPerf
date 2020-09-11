#!/bin/bash
MODEL_DIR=../output
rm -rf $MODEL_DIR
gpus=${1:-"0"}
bz_per_device=${2:-32}
DTYPE=${3:-'fp32'}
NUM_STEP=${4:-120}
TEST_NUM=${5:-1}

a=`expr ${#gpus} + 1`
NUM_GPU=`expr ${a} / 2`
total_batch_size=`expr ${bz_per_device} \* $NUM_GPU`
echo "Use gpus: $gpus"
echo "Total batch size : $total_batch_size"


BERT_BASE_CONFIG_FILE='/datasets/bert/uncased_L-12_H-768_A-12/bert_config.json' 
LOG_FOLDER=./logs/tensorflow/bert/1n${NUM_GPU}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/bert_b${bz_per_device}_${DTYPE}_${TEST_NUM}.log


export CUDA_VISIBLE_DEVICES=$gpus
python run_pretraining.py  \
--input_files='/datasets/bert/wiki/*.tfrecord'   \
--max_seq_length=128  \
--max_predictions_per_seq=20  \
--train_batch_size=$total_batch_size  \
--num_steps_per_epoch=$NUM_STEP  \
--num_train_epochs=1 \
--warmup_steps=10000  \
--use_next_sentence_label=True  \
--train_summary_interval=0 \
--optimizer_type='adamw' \
--num_gpus=$NUM_GPU  \
--datasets_num_private_threads=8 \
--dtype=$DTYPE   \
--enable_xla='false' \
--model_dir=$MODEL_DIR   \
--bert_config_file=${BERT_BASE_CONFIG_FILE}   2>&1 | tee $LOGFILE

echo "Writting log to $LOGFILE"