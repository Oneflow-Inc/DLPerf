rm -rf core.*

suffix=$1
DATA_ROOT=/data/wdl_ofrecord
BENCH_ROOT=/OneFlow-Benchmark/ClickThroughRate/WideDeepLearning

log_root=./log
VOCAB_SIZE=2322444 

test_case=${log_root}/n1g1-500iters-$suffix
oneflow_log_file=${test_case}.log

export PYTHONUNBUFFERED=1
python3 $BENCH_ROOT/wdl_train_eval.py \
  --gpu_num 1 \
  --num_nodes 1 \
  --hidden_units_num 2 \
  --train_data_part_num 256 \
  --train_part_name_suffix_length=5 \
  --eval_data_part_num 256 \
  --eval_part_name_suffix_length=5 \
  --max_iter=500 \
  --loss_print_every_n_iter=1 \
  --eval_interval=1 \
  --batch_size=512 \
  --deep_embedding_vec_size 16 \
  --wide_vocab_size=$VOCAB_SIZE \
  --deep_vocab_size=$VOCAB_SIZE > $oneflow_log_file

  #--train_data_dir $DATA_ROOT/train \
  #--eval_data_dir $DATA_ROOT/val \
