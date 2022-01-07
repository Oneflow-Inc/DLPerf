hugectr_root=/path/to/hugectr/test/scripts
data_root=/path/to/criteo

docker run \
  --runtime=nvidia --rm -it \
  --shm-size=16g --ulimit memlock=-1 \
  --net=host \
  --privileged \
  --cap-add=IPC_LOCK --device=/dev/infiniband \
  -v $hugectr_root:/workspace \
  -v $data_root:/workspace/criteo \
  -w /workspace \
  hugectr:devel bash

