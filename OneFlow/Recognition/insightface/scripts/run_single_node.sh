# !/bin/bash
export PYTHONUNBUFFERED=1

workspace=${1:-"/oneflow_face"}
network=${2:-"r100"}
dataset=${3:-"emore"}
loss=${4:-"arcface"}
num_nodes=${5:-1}
bz_per_device=${6:-64}
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

i=1
while [ $i -le 5 ]; do
  bash ${workspace}/scripts/train_insightface.sh ${workspace} ${network} ${dataset} ${loss} 1 ${bz_per_device} ${train_unit} ${train_iter} 1 ${precision} ${model_parallel} ${partial_fc} $i ${sample_ratio} ${num_classes} ${use_synthetic_data}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 5 ]; do
  bash ${workspace}/scripts/train_insightface.sh ${workspace} ${network} ${dataset} ${loss} 1 ${bz_per_device} ${train_unit} ${train_iter} 4 ${precision} ${model_parallel} ${partial_fc} $i ${sample_ratio} ${num_classes} ${use_synthetic_data}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 5 ]; do
  bash ${workspace}/scripts/train_insightface.sh ${workspace} ${network} ${dataset} ${loss} 1 ${bz_per_device} ${train_unit} ${train_iter} 8 ${precision} ${model_parallel} ${partial_fc} $i ${sample_ratio} ${num_classes} ${use_synthetic_data}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
