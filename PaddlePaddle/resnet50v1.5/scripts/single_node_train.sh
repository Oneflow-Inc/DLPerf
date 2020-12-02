MODEL=${1:-"resnet50"}
gpus=${2:-"0,1,2,3,4,5,6,7"}
BATCH_SIZE=${3:-128}
IMAGE_SIZE=${4:-224}
TEST_NUM=${5:-1}
DTYPE=${6:-"fp32"}

a=`expr ${#gpus} + 1`
GPU_COUNT=`expr ${a} / 2`
total_bz=`expr ${BATCH_SIZE} \* ${GPU_COUNT}`
LR=$(awk -v total_bz="$total_bz" 'BEGIN{print  total_bz / 1000}')


LOG_FOLDER=../logs/paddle/resnet50/bz${BATCH_SIZE}/1n${GPU_COUNT}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/r50_b${BATCH_SIZE}_${DTYPE}_${TEST_NUM}.log


DATA_DIR=/datasets/ImageNet/Paddle

MULTI_PROCESS="-m paddle.distributed.launch"
if  [ $GPU_COUNT -le 2 ] ; then
  THREAD=8
elif  [ $GPU_COUNT -le 4 ] ; then
  THREAD=12
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


USE_DALI=false
if ${USE_DALI}; then
    export FLAGS_fraction_of_gpu_memory_to_use=0.8
    export DALI_EXTRA_PATH=/home/leinao/paddle/DALI_extra
    THREAD=10
else
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
fi
echo "FLAGS_fraction_of_gpu_memory_to_use=$FLAGS_fraction_of_gpu_memory_to_use"


echo "Use gpus: $gpus, Batch size per device : $BATCH_SIZE, Total Batch size : $total_bz"
echo "Learning rate: $LR"
# echo "Use fp16 : $use_fp16"

export CUDA_VISIBLE_DEVICES=${gpus}
python3   $MULTI_PROCESS \
        train.py  ${FP16_PARAMS} \
       --data_format=${DATA_FORMAT} \
        --data_dir=${DATA_DIR} \
        --total_images=1302936 \
        --class_dim=1000 \
        --validate=False \
        --model="ResNet50"  \
        --batch_size=${total_bz} \
		    --print_step=1 \
	      --save_step=10000 \
        --reader_thread=${THREAD}   \
        --lr_strategy=cosine_decay \
        --lr=0.001 \
        --momentum_rate=0.875 \
        --image_shape 3 $IMAGE_SIZE $IMAGE_SIZE \
        --max_iter=120 \
        --model_save_dir=output/ \
        --l2_decay=0.000030518 \
     		--warm_up_epochs=1 \
        --use_mixup=False \
        --use_label_smoothing=True \
        --use_dali=$USE_DALI \
        --label_smoothing_epsilon=0.1  2>&1 | tee ${LOGFILE}
echo "Writting log to ${LOGFILE}"
