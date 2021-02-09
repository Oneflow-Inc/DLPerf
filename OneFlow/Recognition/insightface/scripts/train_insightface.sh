export ONEFLOW_DEBUG_MODE=""

workspace=${1:-""}
network=${2:-"r100"}
dataset=${3:-"emore"}
loss=${4:-"arcface"}
num_nodes=${5:-1}
batch_size_per_device=${6:-64}
train_unit=${7:-"batch"}
train_iter=${8:-150} 
gpu_num_per_node=${9:-8}
precision=${10:-fp16}
model_parallel=${11:-0}
partial_fc=${12:-0}
test_times=${13:-1}
sample_ratio=${14:-0.1}
num_classes=${15:-1500000}
use_synthetic_data=${16:-False}

MODEL_SAVE_DIR=${num_classes}_${precision}_b${batch_size_per_device}_oneflow_model_parallel_${model_parallel}_partial_fc_${partial_fc}/${num_nodes}n${gpu_num_per_node}g
LOG_DIR=$MODEL_SAVE_DIR

if [ $gpu_num_per_node -gt 1 ]; then
  if [ $network = "r100"]
    data_part_num=16
  elif [$network = "r100_glint360k"]
    data_part_num=200
  else
    echo "Please modify exact data part num in sample_config.py!"
else
    data_part_num=1
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC=" --use_fp16=True"
elif [ "$precision" = "fp32" ] ; then
   PREC=" --use_fp16=False"
else
   echo "Unknown <precision> argument"
   exit -2
fi

LOG_FILE=${LOG_DIR}/${network}_b${batch_size_per_device}_${precision}_$test_times.log

mkdir -p $MODEL_SAVE_DIR

time=$(date "+%Y-%m-%d %H:%M:%S")
echo $time

CMD="$workspace/insightface_train.py"
CMD+=" --network=${network}"
CMD+=" --dataset=${dataset}"
CMD+=" --loss=${loss}"
CMD+=" --train_batch_size=$(expr $num_nodes '*' $gpu_num_per_node '*' $batch_size_per_device)"
CMD+=" --train_unit=${train_unit}"
CMD+=" --train_iter=${train_iter}"
CMD+=" --device_num_per_node=${gpu_num_per_node}"
CMD+=" --model_parallel=${model_parallel}"
CMD+=" --partial_fc=${partial_fc}"
CMD+=" --sample_ratio=${sample_ratio}"
CMD+=" --log_dir=${LOG_DIR}"
CMD+=" $PREC"
CMD+=" --sample_ratio=${sample_ratio}"
CMD+=" --use_synthetic_data=${use_synthetic_data}"
CMD+=" --num_classes=${num_classes}"
CMD+=" --data_part_num=${data_part_num}"

CMD="python3 $CMD "
set -x
if [ -z "$LOG_FILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOG_FILE

fi
set +x
echo "Writing log to ${LOG_FILE}"
