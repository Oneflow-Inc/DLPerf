WORKSPACE=${1:-"/workspace/rn50v15_tf"}
DATA_DIR=${2:-"/data"}
NUM_GPU=${3:-8}
NUM_STEP=${4:-120}
BATCH_SIZE=${5:-128}
DTYPE=${6:-"fp32"}
NUM_TEST=${7:-1}
OTHER=${@:8}

LOG_FOLDER=../logs/ngc/tensorflow/resnet50/bz${BATCH_SIZE}/1n${NUM_GPU}g
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/r50_b${BATCH_SIZE}_${DTYPE}_$NUM_TEST.log

if [[ ! -z "${BIND_TO_SOCKET}" ]]; then
    BIND_TO_SOCKET="--bind-to socket"
fi

if [[ ! -z "${USE_DALI}" ]]; then
    USE_DALI="--use_dali --data_idx_dir=${DATA_DIR}/dali_idx"
fi

if [[ ! -z "${USE_XLA}" ]]; then
    USE_XLA="--use_xla"
fi

CMD=""
case $DTYPE in
    "fp32") CMD+="--precision=fp32";;
    "fp16") CMD+="--precision=fp16 --use_static_loss_scaling --loss_scale=128";;
    "amp") CMD+="--precision=fp32 --use_tf_amp --use_static_loss_scaling --loss_scale=128";;
esac

CMD="--arch=resnet50 --mode=train --iter_unit=batch --num_iter=${NUM_STEP} \
    --batch_size=${BATCH_SIZE} --warmup_steps=0 --use_cosine_lr --label_smoothing 0.1 \
    --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05 \
    ${CMD} --data_dir=${DATA_DIR}/tfrecords ${USE_DALI} ${USE_XLA} \
    --results_dir=${LOG_FOLDER}/results --weight_init=fan_in ${OTHER} \
    --display_every=1  --gpu_memory_fraction=0.98"

if [[ ${NUM_GPU} -eq 1 ]]; then
    python3 main.py ${CMD}  2>&1 | tee ${LOGFILE}
else
    mpiexec --allow-run-as-root ${BIND_TO_SOCKET} -np ${NUM_GPU} python3 main.py ${CMD}  2>&1 | tee ${LOGFILE}
fi


echo "Writting log to ${LOGFILE}"
