#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

DATA_ROOT=/datasets/output_data
DATA_PATH=/datasets/openwebtext-json/openwebtext
VOCAB_PATH=./gpt2-vocab.json
MERges_PATH=./gpt2-merges.txt

#python tools/preprocess_data_from_txt.py \
python tools/preprocess_data.py \
       --input $DATA_PATH \
       --output-prefix $DATA_ROOT \
       --vocab $VOCAB_PATH \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERges_PATH \
       --append-eod
