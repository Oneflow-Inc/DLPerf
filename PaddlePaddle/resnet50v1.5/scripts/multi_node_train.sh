MODEL=${1:-"resnet50"}
gpus=${2:-"0,1,2,3,4,5,6,7"}
BATCH_SIZE=${3:-128}
IMAGE_SIZE=${4:-224}
nodes=${5:-$NODE1,$NODE2,NODE3,$NODE4}
CURRENT_NODE=${6:-NODE1}
TEST_NUM=${7:-1}
DTYPE=${8:-"fp32"}

a=`expr ${#gpus} + 1`
GPUS_PER_NODE=`expr ${a} / 2`
total_bz=`expr ${BATCH_SIZE} \* ${GPUS_PER_NODE}`
LR=$(awk -v total_bz="$total_bz" 'BEGIN{print  total_bz / 1000}')
node_num=$(echo $nodes | tr ',' '\n' | wc -l)
NUM_EPOCH=`expr ${node_num} \* 4`


LOG_FOLDER=../logs/paddle/resnet50/bz${BATCH_SIZE}/${node_num}n${GPUS_PER_NODE}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/r50_b${BATCH_SIZE}_${DTYPE}_${TEST_NUM}.log


export CUDA_VISIBLE_DEVICES=${gpus}
export FLAGS_fraction_of_gpu_memory_to_use=0.98
DATA_DIR=/datasets/ImageNet/imagenet_1k/


if  [ $node_num -le 2 ] ; then
  THREAD=8
else
  THREAD=8
fi


# bash run.sh train ResNet50_fp16
if  [ "$DTYPE" == "fp16" ] ; then
  export FLAGS_conv_workspace_size_limit=4000 #MB
  export FLAGS_cudnn_exhaustive_search=1
  export FLAGS_cudnn_batchnorm_spatial_persistent=1
  DATA_FORMAT="NHWC"
  FP16_PARAMS="   --use_fp16=True  --use_dynamic_loss_scaling=true  --scale_loss=128.0   --fuse_elewise_add_act_ops=true   --fuse_bn_act_ops=true  "
else
  DATA_FORMAT="NCHW"
  FP16_PARAMS=" "
fi

echo "Nodes : $nodes"
echo "Use gpus: $gpus, Batch size per device : $BATCH_SIZE, Total Batch size : $total_bz"
echo "Learning rate: $LR"
echo "Use fp16 : $use_fp16"


python3 -m paddle.distributed.launch --cluster_node_ips=${nodes} \
--node_ip=$CURRENT_NODE \
train.py \
        $FP16_PARAMS  \
        --data_format=${DATA_FORMAT} \
        --reader_thread=$THREAD \
        --data_dir=${DATA_DIR} \
        --total_images=1302936 \
        --class_dim=1000 \
        --validate=False \
        --batch_size=$total_bz \
        --image_shape 3 $IMAGE_SIZE $IMAGE_SIZE \
      	--print_step=1 \
	      --save_step=10000 \
        --lr_strategy=piecewise_decay \
        --lr=0.001 \
        --momentum_rate=0.875 \
        --max_iter=120 \
        --model='ResNet50'  \
        --model_save_dir=output/ \
        --l2_decay=0.000030518 \
        --warm_up_epochs=1 \
        --use_mixup=False \
        --use_label_smoothing=True \
        --label_smoothing_epsilon=0.1  2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"