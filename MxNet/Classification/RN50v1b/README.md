# MXNet ResNet50v1b  测评

## 概述 Overview

本测试基于 [gluon-cv](https://github.com/dmlc/gluon-cv) 仓库中提供的 MXNet框架的 [ResNet50](https://github.com/dmlc/gluon-cv/blob/f9a8a284b8/scripts/classification/imagenet/README.md) 实现，进行了1机1卡、1机8卡、2机16卡、4机32卡的结果复现及速度评测，得到吞吐率及加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试覆盖了FP32 精度、FP16混合精度，后续将持续维护，增加更多方式的测评。



## 环境 Environment

### 系统

- #### 硬件

  - GPU：8x Tesla V100-SXM2-16GB

- #### 软件

  - 驱动：NVIDIA 440.33.01
  
  - 系统：[ Ubuntu 16.04](http://releases.ubuntu.com/16.04/)
  
  - CUDA：10.2
  
  - cuDNN：7.6.5
  
  - NCCL：2.7.3
  
  - OpenMPI 4.0.0
  
  - Horovod 0.19.5
  
  - Python：3.7.7
  
- #### 框架
  
  - **MXNet 1.6.0** 

- #### Feature support matrix

| Feature | ResNet50-v1b MXNet |
| ----------------------------------- | ------- |
| Horovod/MPI Multi-GPU      |  Yes    |
| Horovod/MPI Multi-Node    | Yes     |
| Automatic mixed precision (AMP) |  Yes    |


## 快速开始 Quick Start

### 1. 前期准备
- #### 数据集
数据集制作方式参考[NVIDIA官方提供的MXNet数据集制作方法](https://github.com/NVIDIA/DeepLearningExamples/tree/e470c2150abf4179f873cabad23945bbc920cc5f/MxNet/Classification/RN50v1.5#getting-the-data)
- #### SSH 免密

  单机测试下无需配置，但测试2机、4机等多机情况下，则需要配置节点间的ssh免密登录，保证MXNet 的 mpi 分布式脚本运行时可以在单机上与其他节点互联。

  

 - #### 环境安装
```shell
# 安装mxnet
python3 -m pip install gluoncv==0.9.0b20200915 autogluon==0.0.14 -i https://mirror.baidu.com/pypi/simple
# 安装horovod（安装前，确保环境中已有nccl、openmpi）
HOROVOD_WITH_MXNET=1  HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL python3 -m pip install --no-cache-dir horovod==0.19.5
```


### 2. 额外准备

- #### 下载gluon-cv仓库源码

  ```shell
  git clone https://github.com/dmlc/gluon-cv.git 
  git checkout f9a8a284b8222794bc842453e2bebe5746516048
  ```

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

将本仓库 DLPerf/MxNet/Classification/RN50v1b/ 路径源码放至 gluon-cv/scripts/classification/imagenet 下，执行脚本

```shell
bash run_test.sh 128 fp32
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

测试进行了多组训练（本测试中取 5 次），每次训练过程只取第 1 个 epoch 的前 120 iter，计算训练速度时去掉前 20 iter，只取后 100 iter 的数据，以降低抖动。最后将 5 次训练的速度取中位数得到最终速度，并最终以此数据计算加速比。

运行，即可得到针对不同配置测试 log 数据处理的结果： 

```shell
python extract_mxnet_logs.py --log_dir=fp32_b128/mxnet/resnet50/bz128 --batch_size_per_device=128
```

结果打印如下：

```shell
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_4.log {4: 740.66}
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_2.log {4: 740.66, 2: 749.04}
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_3.log {4: 740.66, 2: 749.04, 3: 754.4}
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_1.log {4: 740.66, 2: 749.04, 3: 754.4, 1: 755.7}
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_5.log {4: 740.66, 2: 749.04, 3: 754.4, 1: 755.7, 5: 748.58}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_4.log {4: 4855.34}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_2.log {4: 4855.34, 2: 4850.36}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_3.log {4: 4855.34, 2: 4850.36, 3: 4843.76}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_1.log {4: 4855.34, 2: 4850.36, 3: 4843.76, 1: 4924.15}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_5.log {4: 4855.34, 2: 4850.36, 3: 4843.76, 1: 4924.15, 5: 4866.14}
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_4.log {4: 385.31}
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_2.log {4: 385.31, 2: 384.16}
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_3.log {4: 385.31, 2: 384.16, 3: 384.19}
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_1.log {4: 385.31, 2: 384.16, 3: 384.19, 1: 385.23}
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_5.log {4: 385.31, 2: 384.16, 3: 384.19, 1: 385.23, 5: 384.04}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_4.log {4: 9568.3}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_2.log {4: 9568.3, 2: 9611.24}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_3.log {4: 9568.3, 2: 9611.24, 3: 9579.55}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_1.log {4: 9568.3, 2: 9611.24, 3: 9579.55, 1: 9579.74}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_5.log {4: 9568.3, 2: 9611.24, 3: 9579.55, 1: 9579.74, 5: 9631.67}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_4.log {4: 1494.58}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_2.log {4: 1494.58, 2: 1492.14}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_3.log {4: 1494.58, 2: 1492.14, 3: 1501.75}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_1.log {4: 1494.58, 2: 1492.14, 3: 1501.75, 1: 1498.84}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_5.log {4: 1494.58, 2: 1492.14, 3: 1501.75, 1: 1498.84, 5: 1494.29}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_4.log {4: 2564.3}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_2.log {4: 2564.3, 2: 2539.38}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_3.log {4: 2564.3, 2: 2539.38, 3: 2561.71}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_1.log {4: 2564.3, 2: 2539.38, 3: 2561.71, 1: 2556.03}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_5.log {4: 2564.3, 2: 2539.38, 3: 2561.71, 1: 2556.03, 5: 2532.26}
{'rn50': {'1n1g': {'average_speed': 384.59,
                   'batch_size_per_device': 128,
                   'median_speed': 384.19,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 749.68,
                   'batch_size_per_device': 128,
                   'median_speed': 749.04,
                   'speedup': 1.95},
          '1n4g': {'average_speed': 1496.32,
                   'batch_size_per_device': 128,
                   'median_speed': 1494.58,
                   'speedup': 3.89},
          '1n8g': {'average_speed': 2550.74,
                   'batch_size_per_device': 128,
                   'median_speed': 2556.03,
                   'speedup': 6.65},
          '2n8g': {'average_speed': 4867.95,
                   'batch_size_per_device': 128,
                   'median_speed': 4855.34,
                   'speedup': 12.64},
          '4n8g': {'average_speed': 9594.1,
                   'batch_size_per_device': 128,
                   'median_speed': 9579.74,
                   'speedup': 24.93}}}
Saving result to ./result/bz128_result.json
```



### 5. 计算规则

#### 5.1 测速脚本

- extract_mxnet_logs.py

extract_mxnet_logs.py根据官方在log中打印的速度，在120个iter中，排除前20iter，取后100个iter的速度做平均；


#### 5.2 均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度。

#### 5.3 加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5



## 性能结果 Performance

该小节提供针对 MXNet 框架的 ResNet50 v1b 模型单机测试的性能结果和完整 log 日志。

### ResNet50 v1b FP32

#### batch_size = 128 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 384.19    | 1.00    |
| 1        | 2       | 749.04    | 1.95    |
| 1        | 4       | 1494.58   | 3.89    |
| 1        | 8       | 2556.03   | 6.65    |
| 2        | 16      | 4855.34   | 12.64   |
| 4        | 32      | 9579.74   | 24.93   |

### ResNet50 v1b FP16

#### batch_size = 256 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 833.65    | 1       |
| 1        | 2       | 1373.55   | 1.65    |
| 1        | 4       | 2331.91   | 2.8     |
| 1        | 8       | 2908.88   | 3.49    |
| 2        | 16      | 5451.27   | 6.54    |
| 4        | 32      | 10565.55  | 12.67   |


详细 Log 信息可下载：

- [mxnet_resnet50v1b_fp32_128.tar.gz](http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MxNet/resnet50/mxnet_resnet50v1b_fp32_128.tar.gz) 

- [mxnet_resnet50v1b_fp16_b256.tar.gz](http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MxNet/resnet50/mxnet_resnet50v1b_fp16_b256.tar.gz) 
