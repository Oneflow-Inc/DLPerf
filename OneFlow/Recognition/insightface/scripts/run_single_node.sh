# !/bin/bash 

workspace=${1:-"/home/leinao/sx/test_face"}
network=${2:-"r100"}
dataset=${3:-"emore"}
loss=${4:-"arcface"}
bz_per_device=${5:-128}
train_unit=${6:-"batch"}
iter_num=${7:-150}
precision=${8:-fp32}
model_parallel=${9:-True}
partila_fc=${10:-True}
sample_ratio=${11:-0.1}
num_classes=${12:-1500000}
use_synthetic_data=${13:-False}

i=1
while [ $i -le 5 ]
do
  bash ${workspace}/scripts/train_emore.sh ${workspace} ${network} ${dataset} ${loss} 1 ${bz_per_device} ${train_unit} ${iter_num} 1 ${precision} ${model_parallel} ${partila_fc} $i  ${sample_ratio} ${num_classes} ${use_synthetic_data}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash ${workspace}/scripts/train_emore.sh ${workspace} ${network} ${dataset} ${loss} 1 ${bz_per_device} ${train_unit} ${iter_num} 1 ${precision} ${model_parallel} ${partila_fc} $i  ${sample_ratio} ${num_classes} ${use_synthetic_data} 
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 5 ]
do
  bash ${workspace}/scripts/train_emore.sh ${workspace} ${network} ${dataset} ${loss} 1 ${bz_per_device} ${train_unit} ${iter_num} 1 ${precision} ${model_parallel} ${partila_fc} $i  ${sample_ratio} ${num_classes} ${use_synthetic_data}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
