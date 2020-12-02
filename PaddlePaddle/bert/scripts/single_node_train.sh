#!/bin/bash
OUTPUT_DIR=../output
MODEL=${1:-"bert_base"}
gpus=${2:-"0,1,2,3,4,5,6,7"}
BATCH_SIZE=${3:-32}
TEST_NUM=${4:-1}
DTYPE=${5:-"fp32"}

a=`expr ${#gpus} + 1`
NUM_GPU=`expr ${a} / 2`
paddle_batch_size=`expr ${BATCH_SIZE} \* 128`
echo "Use gpus: $gpus"
echo "Batch size per device : $BATCH_SIZE, Paddle Batch size : $paddle_batch_size"


LOG_FOLDER=./logs/paddle/bert/bz${BATCH_SIZE}/1n${NUM_GPU}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/bert_b${BATCH_SIZE}_${DTYPE}_$TEST_NUM.log

export CUDA_VISIBLE_DEVICES=${gpus}
if [ "$MODEL" = "bert_base" ] ; then
    CONFIG_PATH=${BERT_BASE_CONFIG}
    VOCAB_PATH='data/demo_config/vocab.txt'
    max_seq_len=128
    max_predictions_per_seq=20
    PADDLE_BERT_DATA_DIR=$PADDLE_BERT_BASE_DATA_DIR
else
    CONFIG_PATH=${BERT_LARGE_CONFIG}
    VOCAB_PATH=${SCRIPT_ROOT_DIR}/configs/bert_model_config/uncased_L-24_H-1024_A-16/vocab.txt
    max_seq_len=512
    max_predictions_per_seq=80
    PADDLE_BERT_DATA_DIR=$PADDLE_BERT_LARGE_DATA_DIR
fi


if  [ "$DTYPE" == "fp16" ] ; then
  use_fp16=True
  use_dynamic_loss_scaling=True
  init_loss_scaling=128.0
else
  use_fp16=False
  use_dynamic_loss_scaling=False
  init_loss_scaling=128.0
fi


# Change your train arguments:
python -u ./train.py --is_distributed false \
        --use_cuda true \
        --use_fast_executor true \
        --weight_sharing true \
        --batch_size ${paddle_batch_size} \
        --data_dir ${PADDLE_BERT_DATA_DIR:-'data/train'} \
        --validation_set_dir ${PADDLE_BERT_DATA_DIR:-'data/train'} \
        --bert_config_path ${CONFIG_PATH:-'data/demo_config/bert_config.json'} \
        --vocab_path ${VOCAB_PATH} \
        --generate_neg_sample true\
        --save_steps 10000 \
        --learning_rate 1e-4 \
        --weight_decay 0.01 \
        --warmup_steps 120 \
        --num_train_steps 120 \
        --max_seq_len ${max_seq_len} \
        --skip_steps 1 \
        --validation_steps 1000 \
        --use_fp16 ${use_fp16} \
        --use_dynamic_loss_scaling ${use_dynamic_loss_scaling} \
        --init_loss_scaling ${init_loss_scaling}  \
        --verbose true \
        --checkpoints $OUTPUT_DIR/paddle/runtime_output/checkpoints  2>&1 | tee $LOGFILE

echo "Writting log to $LOGFILE"
       
