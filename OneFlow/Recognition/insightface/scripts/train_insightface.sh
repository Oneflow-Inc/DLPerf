#！/bin/bash
#export ONEFLOW_DEBUG_MODE=True
export PYTHONUNBUFFERED=1

workspace=${1:-"/oneflow_face"}
network=${2:-"r100"}
dataset=${3:-"emore"}
loss=${4:-"arcface"}
num_nodes=${5:-4}
batch_size_per_device=${6:-64}
train_unit=${7:-"batch"}
train_iter=${8:-150}
gpu_num_per_node=${9:-8}
precision=${10:-fp32}
model_parallel=${11:-1}
partial_fc=${12:-1}
test_times=${13:-1}
sample_ratio=${14:-0.1}
num_classes=${15:-85744}
use_synthetic_data=${16:-False}

MODEL_SAVE_DIR=${num_classes}_${precision}_b${batch_size_per_device}_oneflow_model_parallel_${model_parallel}_partial_fc_${partial_fc}/${num_nodes}n${gpu_num_per_node}g
LOG_DIR=$MODEL_SAVE_DIR

if [ $gpu_num_per_node -gt 1 ]; then
   if [ $network = "r100" ]; then
      data_part_num=32
   elif [ $network = "r100_glint360k" ]; then
      data_part_num=200
   else
      echo "Please modify exact data part num in sample_config.py!"
   fi
else
   data_part_num=1
fi

sed -i "s/${dataset}.train_data_part_num = [[:digit:]]*/${dataset}.train_data_part_num = $data_part_num/g" $workspace/sample_config.py
sed -i "s/${dataset}.num_classes = [[:digit:]]*/${dataset}.num_classes = $num_classes/g" $workspace/sample_config.py

PREC=""
if [ "$precision" = "fp16" ]; then
   PREC=" --use_fp16=True"
elif [ "$precision" = "fp32" ]; then
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
CMD+=" --num_nodes=${num_nodes}"
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
CMD+=" --iter_num_in_snapshot=5000"
CMD+=" --validation_interval=5000"

CMD="/home/leinao/anaconda3/envs/insightface/bin/python3 $CMD "
set -x
if [ -z "$LOG_FILE" ]; then
   $CMD
else
   (
      $CMD
   ) |& tee $LOG_FILE

fi
set +x
echo "Writing log to ${LOG_FILE}"
