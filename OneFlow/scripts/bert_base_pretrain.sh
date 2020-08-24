NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BSZ_PER_DEVICE=$3 

BENCH_ROOT=BERT
DATA_ROOT=/path/to/bert_base_ofrecord
DATA_PART_NUM=32

if [ -n "$4" ]; then
    NODE_IPS=$4
else
    NODE_IPS='10.11.0.2','10.11.0.3','10.11.0.4','10.11.0.5'
fi

rm -rf ./log
mkdir ./log

export PYTHONUNBUFFERED=1
python3 ./$BENCH_ROOT/run_pretraining.py \
  --gpu_num_per_node=$GPU_NUM_PER_NODE \
  --num_nodes=$NUM_NODES \
  --node_ips=$NODE_IPS \
  --learning_rate=1e-4 \
  --batch_size_per_device=$BSZ_PER_DEVICE \
  --iter_num=140 \
  --loss_print_every_n_iter=20 \
  --seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 \
  --hidden_size_per_head=64 \
  --data_dir=$DATA_ROOT \
  --data_part_num=$DATA_PART_NUM \
  --log_dir=./log \
  --model_save_every_n_iter=10000 \
  --save_last_snapshot=False \
  --model_save_dir=./snapshots
  #--node_ips='10.11.0.2','10.11.0.3','10.11.0.4','10.11.0.5' \
  
