#! /bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
rm -rf /results/*
echo "Container nvidia build = " $NVIDIA_BUILD_ID

DATA_DIR=${1:-"data"}
GPUS_PER_NODE=${2:-8}
train_batch_size=${3:-32}
train_steps=${4:-120}
bert_model="base"
max_pred_per_seq=20
seq_len=128
precision=${5:-"fp32"}
use_xla=${6:-"false"}
NODES=${7:-$NODE1,$NODE2}
TEST_NUM=${8:-1}
num_accumulation_steps=1

node_num=$(echo $NODES | tr ',' '\n' | wc -l)
gpu_num=`expr ${node_num} \* ${GPUS_PER_NODE}`
echo "Nodes : ${NODES}"
echo "Total use: ${gpu_num} gpu"

# DATA_DIR=data/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus

if [ "$bert_model" = "large" ] ; then
    export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json
else
    # export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json
    export BERT_CONFIG=data/bert_config.json
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--use_fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "manual_fp16" ] ; then
   PREC="--manual_fp16"
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$use_xla" = "true" ] ; then
    PREC="$PREC --use_xla"
    echo "XLA activated"
fi

export GBS=$(expr $train_batch_size \* $GPUS_PER_NODE \* $num_accumulation_steps)
printf -v TAG "tf_bert_pretraining_adam_%s_%s_gbs%d" "$bert_model" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results/${TAG}_${DATESTAMP}}
LOG_FOLDER=../logs//ngc/tensorflow/bert/bz${train_batch_size}/${node_num}n${GPUS_PER_NODE}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/bert_b${train_batch_size}_${precision}_$TEST_NUM.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

INPUT_FILES="$DATA_DIR/tfrecord"
EVAL_FILES="$DATA_DIR/tfrecord"

horovod_str="--horovod"
mpi=""


CMD="$mpi python3 /workspace/bert/run_pretraining.py"
CMD+=" --input_files_dir=$INPUT_FILES"
CMD+=" --eval_files_dir=$EVAL_FILES"
CMD+=" --output_dir=$RESULTS_DIR"
CMD+=" --bert_config_file=$BERT_CONFIG"
CMD+=" --do_train=True"
CMD+=" --do_eval=False"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --eval_batch_size=32"
CMD+=" --max_seq_length=$seq_len"
CMD+=" --max_predictions_per_seq=$max_pred_per_seq"
CMD+=" --num_train_steps=$train_steps"
CMD+=" --num_warmup_steps=10000"
CMD+=" --num_accumulation_steps=$num_accumulation_steps"
CMD+=" --save_checkpoints_steps=10000"
CMD+=" --learning_rate=1e-4"
CMD+=" --optimizer_type=adam"
CMD+=" $horovod_str $PREC"
CMD+=" --allreduce_post_accumulation=False"

#Check if all necessary files are available before training
for DIR_or_file in $DATA_DIR $BERT_CONFIG $RESULTS_DIR; do
  if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
     echo "Error! $DIR_or_file directory missing. Please mount correctly"
     exit -1
  fi
done

# echo "-np ${gpu_num}, -H ${NODES}, CMD >>>>>>>>>>>>>>>>>>>> ${CMD}" 
horovodrun  -p 10000 -np $gpu_num -H $NODES   $CMD   2>&1 | tee ${LOGFILE}
echo "Writting log to ${LOGFILE}"