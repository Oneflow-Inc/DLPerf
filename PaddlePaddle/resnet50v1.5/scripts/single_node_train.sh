MODEL=${1:-"resnet50"}
gpus=${2:-"0,1,2,3,4,5,6,7"}
BATCH_SIZE=${3:-128}
IMAGE_SIZE=${4:-224}
TEST_NUM=${5:-1}

a=`expr ${#gpus} + 1`
GPU_COUNT=`expr ${a} / 2`
total_bz=`expr ${BATCH_SIZE} \* ${GPU_COUNT}`
LR=$(awk -v total_bz="$total_bz" 'BEGIN{print  total_bz / 1000}')

echo "Use gpus: $gpus"
echo "Batch size per device : $BATCH_SIZE"
echo "Total Batch size : $total_bz"
echo "Learning rate: $LR"

LOG_FOLDER=../paddle/resnet50/1n${GPU_COUNT}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/r50_b${BATCH_SIZE}_fp32_$TEST_NUM.log


export CUDA_VISIBLE_DEVICES=${gpus}
export FLAGS_fraction_of_gpu_memory_to_use=0.98
DATA_DIR=/datasets/ImageNet/imagenet_1k/

if  [ $GPU_COUNT -ge 2 ] ; then
  THREAD=$GPU_COUNT
  MULTI_PROCESS="-m paddle.distributed.launch"
else 
  THREAD=4
  MULTI_PROCESS=""
fi

python3  $MULTI_PROCESS  \
        train.py \
        --data_dir=${DATA_DIR} \
        --total_images=651468 \
        --class_dim=1000 \
        --validate=False \
        --model="ResNet50"  \
        --batch_size=${total_bz} \
		--print_step=1 \
	    --save_step=10000 \
        --reader_thread=$THREAD \
        --lr_strategy=piecewise_decay \
        --lr=$LR \
        --momentum_rate=0.875 \
        --image_shape 3 $IMAGE_SIZE $IMAGE_SIZE \
        --max_iter=120 \
        --model_save_dir=output/ \
        --l2_decay=0.000030518 \
		--warm_up_epochs=1 \
        --use_mixup=False \
        --use_label_smoothing=True \
        --label_smoothing_epsilon=0.1  2>&1 | tee ${LOGFILE}
echo "Writting log to ${LOGFILE}"