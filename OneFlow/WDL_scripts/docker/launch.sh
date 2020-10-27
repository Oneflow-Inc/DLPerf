docker run --rm -it \
  --privileged \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --net=host \
  --cap-add=IPC_LOCK \
  --device=/dev/infiniband \
  -v /path/to/OneFlow-Benchmark:/OneFlow-Benchmark \
  -v /path/to/DLPerf/OneFlow/WDL_scripts:/workspace \
  -v /path/to/datasets:/data \
  -w /workspace \
  oneflow:WDL bash \
  -c "mkdir -p /run/sshd && /usr/sbin/sshd -p 12395 && bash"

