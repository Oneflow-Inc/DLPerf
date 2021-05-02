workspace=/home/leinao/lyon_test/oneflow_face
network=${1:-"r50"}
dataset=${2:-"emore"}
loss=${3:-"arcface"}
num_nodes=${4:-4}
bz_per_device=${5:-128}
train_unit=${6:-"batch"}
train_iter=${7:-150}
precision=${8:-fp32}
model_parallel=${9:-0}
partial_fc=${10:-0}
test_times=${11:-5}
sample_ratio=${12:-1.0}
num_classes=${13:-85744}
use_synthetic_data=${14:-False}



i=1
while [ $i -le $test_times ]; do
  rm -rf new_models
  bash ${workspace}/scripts/train_insightface.sh ${workspace} ${network} ${dataset} ${loss} 4 ${bz_per_device} ${train_unit} ${train_iter} 8 ${precision} ${model_parallel} ${partial_fc} $i ${sample_ratio} ${num_classes} ${use_synthetic_data}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finished Test Case ${i}! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
