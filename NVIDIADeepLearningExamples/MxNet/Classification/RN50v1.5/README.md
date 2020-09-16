# NVIDIA/DeepLearningExamples MXNet ResNet50 v1.5 测评

## 概述 Overview

本测试基于 [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) 仓库中提供的 MXNet框架的 [ResNet50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/e470c2150abf4179f873cabad23945bbc920cc5f/MxNet/Classification/RN50v1.5) 实现，在 NVIDIA 官方提供的 [MXNet 20.03 NGC 镜像及其衍生容器](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet/tags)中进行单机单卡、单机多卡的结果复现及速度评测，并使用Horovod进行多机（2机、4机）的训练，得到吞吐率及加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试已覆盖 FP32 精度、FP6混合精度，后续将持续维护，增加XLA 等多种方式的测评。



## 环境 Environment

### 系统

- #### 硬件

  - GPU：8x Tesla V100-SXM2-16GB

- #### 软件

  - 驱动：NVIDIA 440.33.01

  - 系统：[ Ubuntu 16.04](http://releases.ubuntu.com/16.04/)

  - CUDA：10.2

  - cuDNN：7.6.5

### NGC 容器 20.03

- 系统：[ Ubuntu 18.04](http://releases.ubuntu.com/18.04/)

- CUDA 10.2.89

- cuDNN 7.6.5

- NCCL：2.5.6

- **MXNet：1.6.0**

- OpenMPI 3.1.4

- DALI 0.19.0

- Horovod 0.19.0

- Python：3.6.9

  更多容器细节请参考 [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html)。

  #### Feature support matrix

  | Feature                                                      | ResNet50 v1.5 MXNet |
  | ------------------------------------------------------------ | ------------------- |
  | Horovod/MPI Multi-GPU                                        | Yes                 |
  | Horovod/MPI Multi-Node                                       | Yes                 |
  | [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) | Yes                 |
  | Automatic mixed precision (AMP)                              | Yes                 |



## 快速开始 Quick Start

### 项目代码

### 1. 前期准备

- #### 数据集

  数据集制作参考[NVIDIA官方提供的MXNet数据集制作方法](https://github.com/NVIDIA/DeepLearningExamples/tree/e470c2150abf4179f873cabad23945bbc920cc5f/MxNet/Classification/RN50v1.5#getting-the-data)

- #### 镜像及容器

  同时，根据 [NVIDIA 官方指导 Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/e470c2150abf4179f873cabad23945bbc920cc5f/MxNet/Classification/RN50v1.5#getting-the-data)下载源码、拉取镜像（本次测试选用的是 NGC 20.03）、搭建容器，进入容器环境。

  ```shell
  git clone https://github.com/NVIDIA/DeepLearningExamples.git
  git checkout e470c2150abf4179f873cabad23945bbc920cc5f
  cd DeepLearningExamples/MxNet/Classification/RN50v1.5/
  
  # 构建项目镜像 
  # DeepLearningExamples/MxNet/Classification/RN50v1.5/目录下
  docker build . -t nvidia_rn50_mx:20.03
  
  # 启动容器
  docker run -it \
  --shm-size=16g --ulimit memlock=-1 --privileged --net=host \
  --cap-add=IPC_LOCK --device=/dev/infiniband \
  --name mxnet_dlperf \
  -v /home/leinao/DLPerf/dataset/mxnet:/data/imagenet/train-val-recordio-passthrough \
  -v /home/leinao/DLPerf/:/DLPerf/ \
  nvidia_rn50_mx:20.03
  ```

- #### SSH 免密

  单机测试下无需配置，但测试 2 机、4 机等多机情况下，则需要配置 docker 容器间的 ssh 免密登录，保证MXNet 的 mpi 分布式脚本运行时可以在单机上与其他节点互联。

   **安装ssh服务端**

  ```shell
  # 在容器内执行
  apt-get update
  apt-get install openssh-server
  ```

  **设置免密登录**

  - 节点间的 /root/.ssh/id_rsa.pub 互相授权，添加到 /root/.ssh/authorized_keys 中；
  - 修改 sshd 中用于 docker 通信的端口号 `vim /etc/ssh/sshd_config`，修改 `Port`；
  - 重启 ssh 服务，`service ssh restart`。

### 2. 额外准备

- #### NGC 20.03 mxnet镜像运行DALI with CUDA 10.2 

  注意： NVIDIA DeepLearningExamples 仓库的MXNet最新脚本用的还是19.07的镜像：

  [`FROM nvcr.io/nvidia/mxnet:19.07-py3`](https://github.com/NVIDIA/DeepLearningExamples/blob/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/MxNet/Classification/RN50v1.5/Dockerfile#L1)

  DLPerf仓库中的测试为了统一环境和驱动、第三方依赖的版本，都用了20.03的镜像。mxnet-20.03的镜像里用了与CUDA 10.2相匹配的DALI 0.19.0，而该容器内的脚本还是19.07镜像里的脚本，直接运行dali-cpu和dali-gpu会报错，因此需要做一些修改：
  
  - 修改[Line34](https://github.com/NVIDIA/DeepLearningExamples/blob/e470c2150abf4179f873cabad23945bbc920cc5f/MxNet/Classification/RN50v1.5/dali.py#L34) ----dali-fuse-decoder，将default=1改为default=0
  
  - 把 [dali.py](https://github.com/NVIDIA/DeepLearningExamples/blob/e470c2150abf4179f873cabad23945bbc920cc5f/MxNet/Classification/RN50v1.5/dali.py) 中的`nvJPEGDecoder`替换成`ImageDecoder`，详见： https://github.com/NVIDIA/DALI/issues/906 

### 3. 运行测试

本次测试集群中有 4 台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有 8 张 V100 显卡， 每张显卡显存 16 G。

- #### 测试

在容器内下载本仓库源码：

````shell
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 DLPerf/NVIDIADeepLearningExamples/MxNet/Classification/RN50v1.5/ 路径源码放至 /workspace/rn50 下，执行脚本

```shell
bash run_test.sh 
```

针对1机1卡、1机8卡、2机16卡、4机32卡， batch_size_per_device = **128**，进行测试。

默认测试FP32、batch size=128，也可以指定其他batch size，如64：
```shell
bash run_test.sh 64
```

#### 混合精度

修改run_test.sh中的DTYPE参数为"fp16"即可，或者运行脚本时指定参数，如：

```shell
bash run_test.sh 256 fp16
```

即可对batch size=256，FP16混合精度的条件进行测试。



### 4. 数据处理

测试进行了多组训练（本测试中取 7 次），每次训练过程只取第 1 个 epoch 的前 120 iter，计算训练速度时去掉前 20 iter，只取后 100 iter 的数据，以降低抖动。最后将 7 次训练的速度取中位数得到最终速度，并最终以此数据计算加速比。

运行，即可得到针对不同配置测试 log 数据处理的结果： 

```shell
python extract_mxnet_logs.py --log_dir=logs/ngc/mxnet/resnet50/bz128 --batch_size_per_device=128
```

结果打印如下：

```shell
logs/ngc/mxnet/resnet50/bz128/4n8g/r50_b128_fp32_1.log {1: 11434.12}
logs/ngc/mxnet/resnet50/bz128/4n8g/r50_b128_fp32_4.log {1: 11434.12, 4: 11305.35}
logs/ngc/mxnet/resnet50/bz128/4n8g/r50_b128_fp32_7.log {1: 11434.12, 4: 11305.35, 7: 11461.68}
logs/ngc/mxnet/resnet50/bz128/4n8g/r50_b128_fp32_2.log {1: 11434.12, 4: 11305.35, 7: 11461.68, 2: 11331.93}
logs/ngc/mxnet/resnet50/bz128/4n8g/r50_b128_fp32_3.log {1: 11434.12, 4: 11305.35, 7: 11461.68, 2: 11331.93, 3: 11429.36}
logs/ngc/mxnet/resnet50/bz128/4n8g/r50_b128_fp32_6.log {1: 11434.12, 4: 11305.35, 7: 11461.68, 2: 11331.93, 3: 11429.36, 6: 11313.33}
logs/ngc/mxnet/resnet50/bz128/4n8g/r50_b128_fp32_5.log {1: 11434.12, 4: 11305.35, 7: 11461.68, 2: 11331.93, 3: 11429.36, 6: 11313.33, 5: 11283.49}
logs/ngc/mxnet/resnet50/bz128/1n8g/r50_b128_fp32_1.log {1: 3008.52}
logs/ngc/mxnet/resnet50/bz128/1n8g/r50_b128_fp32_4.log {1: 3008.52, 4: 3009.46}
logs/ngc/mxnet/resnet50/bz128/1n8g/r50_b128_fp32_7.log {1: 3008.52, 4: 3009.46, 7: 2999.97}
logs/ngc/mxnet/resnet50/bz128/1n8g/r50_b128_fp32_2.log {1: 3008.52, 4: 3009.46, 7: 2999.97, 2: 3001.01}
logs/ngc/mxnet/resnet50/bz128/1n8g/r50_b128_fp32_3.log {1: 3008.52, 4: 3009.46, 7: 2999.97, 2: 3001.01, 3: 2993.87}
logs/ngc/mxnet/resnet50/bz128/1n8g/r50_b128_fp32_6.log {1: 3008.52, 4: 3009.46, 7: 2999.97, 2: 3001.01, 3: 2993.87, 6: 3008.01}
logs/ngc/mxnet/resnet50/bz128/1n8g/r50_b128_fp32_5.log {1: 3008.52, 4: 3009.46, 7: 2999.97, 2: 3001.01, 3: 2993.87, 6: 3008.01, 5: 3006.98}
logs/ngc/mxnet/resnet50/bz128/1n4g/r50_b128_fp32_1.log {1: 1520.55}
logs/ngc/mxnet/resnet50/bz128/1n4g/r50_b128_fp32_4.log {1: 1520.55, 4: 1518.04}
logs/ngc/mxnet/resnet50/bz128/1n4g/r50_b128_fp32_7.log {1: 1520.55, 4: 1518.04, 7: 1517.28}
logs/ngc/mxnet/resnet50/bz128/1n4g/r50_b128_fp32_2.log {1: 1520.55, 4: 1518.04, 7: 1517.28, 2: 1521.26}
logs/ngc/mxnet/resnet50/bz128/1n4g/r50_b128_fp32_3.log {1: 1520.55, 4: 1518.04, 7: 1517.28, 2: 1521.26, 3: 1522.3}
logs/ngc/mxnet/resnet50/bz128/1n4g/r50_b128_fp32_6.log {1: 1520.55, 4: 1518.04, 7: 1517.28, 2: 1521.26, 3: 1522.3, 6: 1517.98}
logs/ngc/mxnet/resnet50/bz128/1n4g/r50_b128_fp32_5.log {1: 1520.55, 4: 1518.04, 7: 1517.28, 2: 1521.26, 3: 1522.3, 6: 1517.98, 5: 1516.09}
logs/ngc/mxnet/resnet50/bz128/1n1g/r50_b128_fp32_1.log {1: 392.24}
logs/ngc/mxnet/resnet50/bz128/1n1g/r50_b128_fp32_4.log {1: 392.24, 4: 393.53}
logs/ngc/mxnet/resnet50/bz128/1n1g/r50_b128_fp32_7.log {1: 392.24, 4: 393.53, 7: 391.77}
logs/ngc/mxnet/resnet50/bz128/1n1g/r50_b128_fp32_2.log {1: 392.24, 4: 393.53, 7: 391.77, 2: 390.09}
logs/ngc/mxnet/resnet50/bz128/1n1g/r50_b128_fp32_3.log {1: 392.24, 4: 393.53, 7: 391.77, 2: 390.09, 3: 392.63}
logs/ngc/mxnet/resnet50/bz128/1n1g/r50_b128_fp32_6.log {1: 392.24, 4: 393.53, 7: 391.77, 2: 390.09, 3: 392.63, 6: 392.85}
logs/ngc/mxnet/resnet50/bz128/1n1g/r50_b128_fp32_5.log {1: 392.24, 4: 393.53, 7: 391.77, 2: 390.09, 3: 392.63, 6: 392.85, 5: 391.58}
logs/ngc/mxnet/resnet50/bz128/1n2g/r50_b128_fp32_1.log {1: 767.39}
logs/ngc/mxnet/resnet50/bz128/1n2g/r50_b128_fp32_4.log {1: 767.39, 4: 764.6}
logs/ngc/mxnet/resnet50/bz128/1n2g/r50_b128_fp32_7.log {1: 767.39, 4: 764.6, 7: 761.98}
logs/ngc/mxnet/resnet50/bz128/1n2g/r50_b128_fp32_2.log {1: 767.39, 4: 764.6, 7: 761.98, 2: 765.98}
logs/ngc/mxnet/resnet50/bz128/1n2g/r50_b128_fp32_3.log {1: 767.39, 4: 764.6, 7: 761.98, 2: 765.98, 3: 763.76}
logs/ngc/mxnet/resnet50/bz128/1n2g/r50_b128_fp32_6.log {1: 767.39, 4: 764.6, 7: 761.98, 2: 765.98, 3: 763.76, 6: 767.85}
logs/ngc/mxnet/resnet50/bz128/1n2g/r50_b128_fp32_5.log {1: 767.39, 4: 764.6, 7: 761.98, 2: 765.98, 3: 763.76, 6: 767.85, 5: 761.6}
logs/ngc/mxnet/resnet50/bz128/2n8g/r50_b128_fp32_1.log {1: 5758.49}
logs/ngc/mxnet/resnet50/bz128/2n8g/r50_b128_fp32_4.log {1: 5758.49, 4: 5755.92}
logs/ngc/mxnet/resnet50/bz128/2n8g/r50_b128_fp32_7.log {1: 5758.49, 4: 5755.92, 7: 5803.52}
logs/ngc/mxnet/resnet50/bz128/2n8g/r50_b128_fp32_2.log {1: 5758.49, 4: 5755.92, 7: 5803.52, 2: 5685.5}
logs/ngc/mxnet/resnet50/bz128/2n8g/r50_b128_fp32_3.log {1: 5758.49, 4: 5755.92, 7: 5803.52, 2: 5685.5, 3: 5717.04}
logs/ngc/mxnet/resnet50/bz128/2n8g/r50_b128_fp32_6.log {1: 5758.49, 4: 5755.92, 7: 5803.52, 2: 5685.5, 3: 5717.04, 6: 5765.72}
logs/ngc/mxnet/resnet50/bz128/2n8g/r50_b128_fp32_5.log {1: 5758.49, 4: 5755.92, 7: 5803.52, 2: 5685.5, 3: 5717.04, 6: 5765.72, 5: 5767.21}
{'r50': {'1n1g': {'average_speed': 392.1,
                  'batch_size_per_device': 128,
                  'median_speed': 392.24,
                  'speedup': 1.0},
         '1n2g': {'average_speed': 764.74,
                  'batch_size_per_device': 128,
                  'median_speed': 764.6,
                  'speedup': 1.95},
         '1n4g': {'average_speed': 1519.07,
                  'batch_size_per_device': 128,
                  'median_speed': 1518.04,
                  'speedup': 3.87},
         '1n8g': {'average_speed': 3003.97,
                  'batch_size_per_device': 128,
                  'median_speed': 3006.98,
                  'speedup': 7.67},
         '2n8g': {'average_speed': 5750.49,
                  'batch_size_per_device': 128,
                  'median_speed': 5758.49,
                  'speedup': 14.68},
         '4n8g': {'average_speed': 11365.61,
                  'batch_size_per_device': 128,
                  'median_speed': 11331.93,
                  'speedup': 28.89}}}
Saving result to ./result/bz128_result.json
```



### 5. 计算规则

#### 5.1 测速脚本

- extract_mxnet_logs.py
- extract_mxnet_logs_time.py

两个脚本略有不同，得到的结果稍有误差：

extract_mxnet_logs.py根据官方在log中打印的速度，在120个iter中，排除前20iter，取后100个iter的速度做平均；

extract_mxnet_logs_time.py根据batch size和120个iter中，排除前20iter，取后100个iter的实际运行时间计算速度。

本Readme展示的是extract_mxnet_logs_time.py的计算结果。

#### 5.2 均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行7次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度。

#### 5.3 加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5



## 性能结果 Performance

该小节提供针对 NVIDIA MXNet 框架的 ResNet50 v1.5 模型单机测试的性能结果和完整 log 日志。

### FP32 & W/O XLA

- ### ResNet50 v1.5 batch_size = 128

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 392.24    | 1.00    |
| 1        | 2       | 764.6     | 1.95    |
| 1        | 4       | 1518.04   | 3.87    |
| 1        | 8       | 3006.98   | 7.67    |
| 2        | 16      | 5758.49   | 14.68   |
| 4        | 32      | 11331.93  | 28.89   |

## FP16 & W/O XLA

- ### ResNet50 v1.5 batch_size = 256

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 1337.1    | 1       |
| 1        | 4       | 4815.16   | 3.60    |
| 1        | 8       | 8816.38   | 6.59    |
| 2        | 16      | 15246.71  | 11.4    |
| 4        | 32      | 28381.23  | 21.23   |

NVIDIA的 MXNet 官方测评结果详见 [ResNet50 v1.5 For MXNet results](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5#training-performance-nvidia-dgx-1-8x-v100-16g)

详细 Log 信息可下载：

- [resnet50_fp32.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/MxNet/cnn/resnet50_fp32.zip) 

- [resnet50_fp16.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/MxNet/cnn/resnet50_fp16.zip) 