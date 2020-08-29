# NVIDIA/DeepLearningExamples Pytorch ResNet50 v1.5 测评

## 概述 Overview

本测试基于 [NVIDIA/DeepLearningExamples/Classification/ConvNets/](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets) 仓库中提供的 Pytorch 框架的 [ResNet50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) 实现，在 NVIDIA 官方提供的 [20.03 NGC 镜像及其衍生容器](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags)中进行单机单卡、单机多卡的结果复现及速度评测，同时根据 Pytorch 官方的分布式实现，添加 DALI 数据加载方式，测试 1 机、2 机、4 机的吞吐率及加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试仅覆盖 FP32 精度，后续将持续维护，增加混合精度训练，XLA 等多种方式的测评。

## 内容目录 Table Of Content

- [NVIDIA/DeepLearningExamples Pytorch ResNet50 v1.5 测评](#nvidia-deeplearningexamples-pytorch-resnet50-v15---)
  * [概述 Overview](#---overview)
  * [内容目录 Table Of Content](#-----table-of-content)
  * [环境 Environment](#---environment)
    + [系统](#--)
      - [硬件](#--)
    + [NGC 容器](#ngc---)
      - [Feature support matrix](#feature-support-matrix)
  * [快速开始 Quick Start](#-----quick-start)
    + [项目代码](#----)
    + [1. 前期准备](#1-----)
      - [数据集](#---)
      - [镜像及容器](#-----)
      - [SSH 免密](#ssh---)
    + [2. 运行测试](#2-----)
      - [单机测试](#----)
    + [3. 数据处理](#3-----)
  * [性能结果 Performance](#-----performance)
    + [FP32 & W/O XLA](#fp32---w-o-xla)
    + [ResNet50 v1.5 batch_size = 128](#resnet50-v15-batch-size---128)

## 环境 Environment

### 系统

- #### 硬件

  - GPU：Tesla V100（16G）×8

- ####　软件

  - 驱动：Nvidia 440.33.01

  - 系统：[ Ubuntu 16.04](http://releases.ubuntu.com/16.04/)

  - CUDA：10.2

  - cuDNN：7.6.5

### NGC 容器

- 系统：[ Ubuntu 18.04](http://releases.ubuntu.com/18.04/)

- CUDA 10.2.89

- cuDNN 7.6.5

- NCCL：2.5.6

- Pytorch：1.5.0a0+8f84ded

- OpenMPI 3.1.4

- DALI 0.19.0

- Python：3.6.9

  更多容器细节请参考 [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html)。

  #### Feature support matrix

  | Feature                                                      | ResNet50 v1.5 Pytorch |
  | ------------------------------------------------------------ | --------------------- |
  | Multi-gpu training                                           | Yes                   |
  | Multi-node training                                          | Yes                   |
  | [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) | Yes                   |
  | NVIDIA NCCL                                                  | Yes                   |
  | Automatic mixed precision (AMP)                              | No                    |



## 快速开始 Quick Start

### 项目代码

### 1. 前期准备

- #### 数据集

根据 [Convolutional Networks for Image Classification in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets) 准备 ImageNet 数据集，只需下载、解压 train、validation 数据集到对应路径即可，使用原始图片进行训练。

同时，根据该指导下载源码及镜像、搭建容器，进入容器环境。

- #### 镜像及容器

  同时，根据 [NVIDIA 官方指导 Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide)下载源码、拉取镜像（本次测试选用的是 NGC 20.03）、搭建容器，进入容器环境。

  ```
  git clone https://github.com/NVIDIA/DeepLearningExamples.git
  cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
  
  # 构建项目镜像 
  # DeepLearningExamples/PyTorch/LanguageModeling/BERT目录下
  docker build . -t nvidia_rn50_pt:20.03-resnet
  # 启动容器
  docker  run -it --shm-size=16g --ulimit memlock=-1 --privileged  \
  --name pt_bert  --net host \
  --cap-add=IPC_LOCK --device=/dev/infiniband \
  -v ./data:/data/ \
  -d nvidia_rn50_pt:20.03
  ```

- #### SSH 免密

  单机测试下无需配置，但测试 2 机、4 机等多机情况下，则需要配置 docker 容器间的 ssh 免密登录，保证 Pytorch 官方的 mpi/nccl 分布式脚本运行时可以在单机上与其他节点互联。

   **安装ssh服务端**

  ```
  # 在容器内执行
  apt-get update
  apt-get install openssh-server
  ```

  **设置免密登录**

  - 节点间的 /root/.ssh/id_rsa.pub 互相授权，添加到 /root/.ssh/authorized_keys 中；
  - 修改 sshd 中用于 docker 通信的端口号 `vim /etc/ssh/sshd_config`，修改 `Port`；
  - 重启 ssh 服务，`service ssh restart`。

### 2. 运行测试

本次测试集群中有 4 台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有 8 张 V100 显卡， 每张显卡显存 16 G。

- #### 单机测试

在节点 1 的容器内下载本仓库源码：

````
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 /DLPerf/NVIDIADeepLearningExamples/Pytorch/resnet50v1.5/scripts 路径源码放至 /workspace/rn50 下，执行脚本

```
bash scripts/run_single_node.sh
```

针对单机单卡、单机2卡、单机4卡、单机8卡， batch_size 取 128 等情况进行分别测试，并将 log 信息保存在当前目录的 /ngc/pytorch/ 对应分布式配置路径中。

### 3. 数据处理

测试进行了多组训练（本测试中取 5 次），每次训练过程只取第 1 个 epoch 的前 120 iter，计算训练速度时去掉前 20 iter，只取后 100 iter 的数据，以降低抖动。最后将 5~7 次训练的速度取中位数得到最终速度，并最终以此数据计算加速比。

运行 /DLPerf/NVIDIADeepLearningExamples/Pytorch/BERT/extract_pytorch_logs_time.py，即可得到针对不同配置测试结果 log 数据处理的结果： 

```
python extract_pytorch_logs_time.py --log_dir [log_dir]
```

结果打印如下

```
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_2.log {2: 369.95}
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_4.log {2: 369.95, 4: 369.84}
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_3.log {2: 369.95, 4: 369.84, 3: 369.48}
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_5.log {2: 369.95, 4: 369.84, 3: 369.48, 5: 369.18}
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_1.log {2: 369.95, 4: 369.84, 3: 369.48, 5: 369.18, 1: 370.22}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_2.log {2: 2861.8}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_7_9.log {2: 2861.8, 7: 690.53}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_4_6.log {2: 2861.8, 7: 690.53, 4: 684.0}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_1_3.log {2: 2861.8, 7: 690.53, 4: 684.0, 1: 674.18}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_10_12.log {2: 2861.8, 7: 690.53, 4: 684.0, 1: 674.18, 10: 702.77}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_4.log {2: 2861.8, 7: 690.53, 4: 2835.59, 1: 674.18, 10: 702.77}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_3.log {2: 2861.8, 7: 690.53, 4: 2835.59, 1: 674.18, 10: 702.77, 3: 2828.1}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_5.log {2: 2861.8, 7: 690.53, 4: 2835.59, 1: 674.18, 10: 702.77, 3: 2828.1, 5: 2833.19}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_13_15.log {2: 2861.8, 7: 690.53, 4: 2835.59, 1: 674.18, 10: 702.77, 3: 2828.1, 5: 2833.19, 13: 676.15}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_1.log {2: 2861.8, 7: 690.53, 4: 2835.59, 1: 2834.07, 10: 702.77, 3: 2828.1, 5: 2833.19, 13: 676.15}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_2.log {2: 1462.66}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_4.log {2: 1462.66, 4: 1462.36}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_3.log {2: 1462.66, 4: 1462.36, 3: 1459.57}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_5.log {2: 1462.66, 4: 1462.36, 3: 1459.57, 5: 1463.17}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_1.log {2: 1462.66, 4: 1462.36, 3: 1459.57, 5: 1463.17, 1: 1463.61}
{'r50': {'1n1g': {'average_speed': 369.73,
                  'batch_size_per_device': 128,
                  'median_speed': 369.84,
                  'speedup': 1.0},
         '1n4g': {'average_speed': 1462.27,
                  'batch_size_per_device': 128,
                  'median_speed': 1462.66,
                  'speedup': 3.95},
         '1n8g': {'average_speed': 2032.78,
                  'batch_size_per_device': 128,
                  'median_speed': 2830.64,
                  'speedup': 7.65}}}
Saving result to ./result/_result.json
```



## 性能结果 Performance

该小节提供针对 NVIDIA Pytorch 框架的 ResNet50 v1.5 模型单机测试的性能结果和完整 log 日志。

### FP32 & W/O XLA

- ### ResNet50 v1.5 batch_size = 128

| gpu_num_per_node | batch_size_per_device | samples/s(Pytorch) | speedup |
| ---------------- | --------------------- | ------------------ | ------- |
| 1                | 128                   | 369.84             | 1.00    |
| 4                | 128                   | 1462.66            | 3.91    |
| 8                | 128                   | 2830.64            | 7.56    |

NVIDIA的 Pytorch 官方测评结果详见 [ResNet50 v1.5 For PyTorch 的 results](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md#results)。

Ray 的 Pytorch 官方测评结果详见 [Distributed PyTorch](https://docs.ray.io/en/master/raysgd/raysgd_pytorch.html#benchmarks).

详细 Log 信息可下载：[ngc_pytorch_resnet50_v1.5.tar]()

