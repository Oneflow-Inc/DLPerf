ONEFLOW_BENCHMARK_ROOT=/path/to/OneFlow-Benchmark
DLPERF_WDL_SCRIPTS_ROOT=/path/to/DLPerf/OneFlow/ClickThroughRate/WideDeepLearning/scripts
DATASET_ROOT=/path/to/datasets:/data

docker run --rm -it \
  --privileged \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --net=host \
  --cap-add=IPC_LOCK \
  --device=/dev/infiniband \
  -v ${ONEFLOW_BENCHMARK_ROOT}:/OneFlow-Benchmark \
  -v ${DLPERF_WDL_SCRIPTS_ROOT}:/workspace \
  -v ${DATASET_ROOT}:/data \
  -w /workspace \
  oneflow:WDL bash \
  -c "mkdir -p /run/sshd && /usr/sbin/sshd -p 12395 && bash"

