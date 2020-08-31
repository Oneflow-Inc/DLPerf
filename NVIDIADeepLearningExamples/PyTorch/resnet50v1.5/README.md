# NVIDIA/DeepLearningExamples PyTorch ResNet50 v1.5 测评

## 概述 Overview

本测试基于 [NVIDIA/DeepLearningExamples/Classification/ConvNets/](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/Classification/ConvNets) 仓库中提供的 PyTorch 框架的 [ResNet50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/Classification/ConvNets/resnet50v1.5) 实现，在 NVIDIA 官方提供的 [20.03 NGC 镜像及其衍生容器](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags)中进行单机单卡、单机多卡的结果复现及速度评测，评判框架在分布式训练情况下的横向拓展能力。

目前，该测试仅覆盖 FP32 精度，后续将持续维护，增加混合精度训练，XLA 等多种方式的测评。

## 内容目录 Table Of Content

* [概述 Overview](#---overview)
* [内容目录 Table Of Content](#-----table-of-content)
* [环境 Environment](#---environment)
  + [系统](#--)
    - [硬件](#--)
    - [软件](#--)
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

  - GPU：Tesla V100-SXM2-16GB x 8

- #### 软件

  - 驱动：NVIDIA 440.33.01

  - 系统：[ Ubuntu 16.04](http://releases.ubuntu.com/16.04/)

  - CUDA：10.2

  - cuDNN：7.6.5

### NGC 容器

- 系统：[ Ubuntu 18.04](http://releases.ubuntu.com/18.04/)

- CUDA 10.2.89

- cuDNN 7.6.5

- NCCL：2.5.6

- PyTorch：1.5.0a0+8f84ded

- OpenMPI 3.1.4

- DALI 0.19.0

- Python：3.6.9

  更多容器细节请参考 [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html)。

  #### Feature support matrix

  | Feature                                                      | ResNet50 v1.5 PyTorch |
  | ------------------------------------------------------------ | --------------------- |
  | Multi-gpu training                                           | Yes                   |
  | [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) | Yes                   |
  | Automatic mixed precision (AMP)                              | No                    |



## 快速开始 Quick Start

### 1. 前期准备

- #### 数据集

根据 [Convolutional Networks for Image Classification in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/Classification/ConvNets) 准备 ImageNet 数据集，只需下载、解压 train、validation 数据集到对应路径即可，使用原始图片进行训练。

- #### 镜像及容器

  拉取 NGC 20.03 的镜像、搭建容器，进入容器环境。

  ```
  # 下载镜像
  docker pull nvcr.io/nvidia/pytorch:20.03-py3 
  
  # 启动容器
  docker run -it --shm-size=16g --ulimit memlock=-1 --privileged  \
  --name pt_bert  --net host \
  --cap-add=IPC_LOCK --device=/dev/infiniband \
  -v ./data:/data/ \
  -d pytorch:20.03-py3 
  ```

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

将本仓库 /DLPerf/NVIDIADeepLearningExamples/PyTorch/resnet50v1.5/scripts 路径源码放至 /workspace/rn50 下，执行脚本

```
bash scripts/run_single_node.sh
```

针对单机单卡、2卡、4卡、8卡， batch_size 取 128 等情况进行测试，并将 log 信息保存在当前目录的 /ngc/pytorch/ 对应分布式配置路径中。

### 3. 数据处理

测试进行了多组训练（本测试中取 5 次），每次训练过程取第 1 个 epoch 的前 120 iter，计算训练速度时只取后 100 iter 的数据，以降低抖动。最后将 5 次训练的结果取中位数得到最终速度，并最终以此数据计算加速比。

运行 /DLPerf/NVIDIADeepLearningExamples/PyTorch/BERT/extract_pytorch_logs_time.py，即可得到针对不同配置测试结果 log 数据处理的结果： 

```
python extract_pytorch_logs_time.py --log_dir /workspace/rn50/scripts/ngc/pytorch --warmup_batches 20 --train_batches 120 --batch_size_per_device 128
```

结果打印如下

```
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_2.log {2: 366.27}
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_4.log {2: 366.27, 4: 366.14}
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_3.log {2: 366.27, 4: 366.14, 3: 365.81}
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_5.log {2: 366.27, 4: 366.14, 3: 365.81, 5: 365.51}
/workspace/rn50/scripts/ngc/pytorch/1n1g/r50_b128_fp32_1.log {2: 366.27, 4: 366.14, 3: 365.81, 5: 365.51, 1: 366.54}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_2.log {2: 2833.72}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_7_9.log {2: 2833.72, 7: 688.8}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_4_6.log {2: 2833.72, 7: 688.8, 4: 682.28}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_1_3.log {2: 2833.72, 7: 688.8, 4: 682.28, 1: 672.63}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_10_12.log {2: 2833.72, 7: 688.8, 4: 682.28, 1: 672.63, 10: 701.07}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_4.log {2: 2833.72, 7: 688.8, 4: 2808.11, 1: 672.63, 10: 701.07}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_3.log {2: 2833.72, 7: 688.8, 4: 2808.11, 1: 672.63, 10: 701.07, 3: 2800.71}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_5.log {2: 2833.72, 7: 688.8, 4: 2808.11, 1: 672.63, 10: 701.07, 3: 2800.71, 5: 2803.56}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_13_15.log {2: 2833.72, 7: 688.8, 4: 2808.11, 1: 672.63, 10: 701.07, 3: 2800.71, 5: 2803.56, 13: 674.42}
/workspace/rn50/scripts/ngc/pytorch/1n8g/r50_b128_fp32_1.log {2: 2833.72, 7: 688.8, 4: 2808.11, 1: 2806.66, 10: 701.07, 3: 2800.71, 5: 2803.56, 13: 674.42}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_2.log {2: 1448.06}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_4.log {2: 1448.06, 4: 1447.61}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_3.log {2: 1448.06, 4: 1447.61, 3: 1445.03}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_5.log {2: 1448.06, 4: 1447.61, 3: 1445.03, 5: 1448.58}
/workspace/rn50/scripts/ngc/pytorch/1n4g/r50_b128_fp32_1.log {2: 1448.06, 4: 1447.61, 3: 1445.03, 5: 1448.58, 1: 1449.08}
{'r50': {'1n1g': {'average_speed': 366.05,
                  'batch_size_per_device': 128,
                  'median_speed': 366.14,
                  'speedup': 1.0},
         '1n4g': {'average_speed': 1447.67,
                  'batch_size_per_device': 128,
                  'median_speed': 1448.06,
                  'speedup': 3.95},
         '1n8g': {'average_speed': 2014.63,
                  'batch_size_per_device': 128,
                  'median_speed': 2802.14,
                  'speedup': 7.65}}}
Saving result to ./result/_result.json
```



## 性能结果 Performance

该小节提供针对 NVIDIA PyTorch 框架的 ResNet50 v1.5 模型单机测试的性能结果和完整 log 日志。

### FP32 & W/O XLA

- ### ResNet50 v1.5 batch_size = 128

| gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
| ---------------- | --------------------- | ------------------ | ------- |
| 1                | 128                   | 366.14             | 1.00    |
| 4                | 128                   | 1448.06            | 3.95    |
| 8                | 128                   | 2802.14            | 7.65    |

NVIDIA的 PyTorch 官方测评结果详见 [ResNet50 v1.5 For PyTorch 的 Results](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/Classification/ConvNets/resnet50v1.5#results)。

Ray 的 PyTorch 官方测评结果详见 [Distributed PyTorch](https://docs.ray.io/en/master/raysgd/raysgd_pytorch.html#benchmarks).

详细 Log 信息可下载：[ngc_pytorch_resnet50_v1.5.tar](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/PyTorch/ngc_pytorch_resnet50_v1.5.tar)
