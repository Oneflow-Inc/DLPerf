#!/usr/bin/bash
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
BZ_PER_DEVICE=${1:-32}


export WORKSPACE=/home/leinao/lyon_test/gluon-nlp/scripts/bert
export DATA_DIR=/home/leinao/DLPerf/dataset/bert_mxnet_npy/wiki_128_npy_part_0

export NODE1=10.11.0.2
export NODE2=10.11.0.3
export NODE3=10.11.0.4
export NODE4=10.11.0.5
export PORT=22
echo "BZ_PER_DEVICE >> ${BZ_PER_DEVICE}"


i=1
while [ $i -le 7 ]
do
    rm -rf ckpt_dir
    bash $SHELL_FOLDER/pretrain.sh bert_base ${BZ_PER_DEVICE} 200 0  1 float32 ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le 7 ]
do
    rm -rf ckpt_dir
    bash $SHELL_FOLDER/pretrain.sh bert_base ${BZ_PER_DEVICE} 200 0,1  1 float32 ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le 7 ]
do
    rm -rf ckpt_dir
    bash $SHELL_FOLDER/pretrain.sh bert_base ${BZ_PER_DEVICE} 200 0,1,2,3  1 float32 ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le 7 ]
do
    rm -rf ckpt_dir
    bash $SHELL_FOLDER/pretrain.sh bert_base ${BZ_PER_DEVICE} 200 0,1,2,3,4,5,6,7  1 float32 ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le 7 ]
do
    rm -rf ckpt_dir
    bash $SHELL_FOLDER/pretrain.sh bert_base ${BZ_PER_DEVICE} 200 0,1,2,3,4,5,6,7 2 float32 ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done


i=1
while [ $i -le 7 ]
do
    rm -rf ckpt_dir
    bash $SHELL_FOLDER/multi_node_pretrain.sh bert_base ${BZ_PER_DEVICE} 200 0,1,2,3,4,5,6,7 4 float32 ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done