#!/bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID
train_batch_size=${1:-48}
num_nodes=${2:-1}
master_node=${3:-127.0.0.1}
master_port=${4:-29500}
num_gpus=${5:-1}
train_steps=${6:-120}
precision=${7:-"fp32"}
test_times=${8:-1}
learning_rate=${9:-"6e-3"}
DATASET=workspace/examples/bert/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
CODEDIR=${10:-"/workspace/examples/bert"}
LOGDIR=./${precision}_ngc_bert_b${train_batch_size}/pytorch/${num_nodes}n${num_gpus}g
CHECKPOINTS_DIR=${CODEDIR}/results/checkpoints
job_name=${11:-"bert-base-adam-training"}
seed=${12:-42}
accumulate_gradients=${13:-"false"}
allreduce_post_accumulation=${14:-"false"}
gradient_accumulation_steps=${15:-1}
allreduce_post_accumulation_fp16=${16:-"false"}
DATA_DIR_PHASE=${17:-$BERT_PREP_WORKING_DIR/${DATASET}/}
resume_training=${18:-"false"}
create_logfile=${19:-"true"}
warmup_proportion=${20:-"1"}
save_checkpoint_steps=${21:-1000}
init_checkpoint=${22:-"None"}
BERT_CONFIG=${CODEDIR}/bert_config.json
mkdir -p $LOGDIR
mkdir -p $CHECKPOINTS_DIR

if [ ! -d "$DATA_DIR_PHASE" ] ; then
   echo "Warning! $DATA_DIR_PHASE directory missing. Training cannot start"
fi
if [ ! -d "$LOGDIR" ] ; then
   echo "Error! $LOGDIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $LOGDIR instead."
   CHECKPOINTS_DIR=$LOGDIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi
ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE
INPUT_DIR=$DATA_DIR_PHASE
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-base-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" --do_train"
CMD+=" --json-summary ${CODEDIR}/dllogger.json "

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus --nnodes $num_nodes --node_rank=0  --master_addr=$master_node  --master_port=$master_port $CMD"
if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pytorch_bert_pretraining_phase_%s_gbs%d" "$precision" $GBS
  LOGFILE=$LOGDIR/${job_name}_b${train_batch_size}_${precision}_${test_times}.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

# in order to test continuously
rm -rf $CHECKPOINTS_DIR

echo "finished training"
