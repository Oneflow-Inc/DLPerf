# !/bin/bash 

workspace=${1:-"/home/leinao/sx/test_face"}
network=${2:-"r100"}
dataset=${3:-"emore"}
loss=${4:-"arcface"}
bz_per_device=${5:-115}
train_unit=${6:-"batch"}
iter_num=${7:-150}
precision=${8:-fp32}
model_parallel=${9:-True}
partila_fc=${10:-False}
sample_ratio=${11:-0.1}

i=1
while [ $i -le 5 ]
do
  bash ${workspace}/scripts/train_glint360k.sh ${workspace} ${network} ${dataset} ${loss} 1 ${bz_per_device} ${train_unit} ${iter_num} 1 ${precision} ${model_parallel} ${partila_fc} $i ${sample_ratio}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 5 ]
do
  bash ${workspace}/scripts/train_glint360k.sh ${workspace} ${network} ${dataset} ${loss} 1 ${bz_per_device} ${train_unit} ${iter_num} 4 ${precision} ${model_parallel} ${partila_fc} $i ${sample_ratio}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 5 ]
do
  bash ${workspace}/scripts/train_glint360k.sh ${workspace} ${network} ${dataset} ${loss} 1 ${bz_per_device} ${train_unit} ${iter_num} 8 ${precision} ${model_parallel} ${partila_fc} $i ${sample_ratio}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
