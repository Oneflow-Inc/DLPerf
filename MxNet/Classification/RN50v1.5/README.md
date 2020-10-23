# MXNet ResNet50  测评

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

| Feature | ResNet50-v1.5 MXNet |
| ----------------------------------- | ------- |
| Horovod/MPI Multi-GPU      |  Yes    |
| Horovod/MPI Multi-Node    | Yes     |
| Automatic mixed precision (AMP) |  Yes    |


## 快速开始 Quick Start

### 1. 前期准备

  
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

将本仓库 DLPerf/MxNet/Classification/RN50v1.5/ 路径源码放至 gluon-cv/scripts/classification/imagenet 下，执行脚本

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
bash run_test.sh 128 fp16
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
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_1.log {1: 394.42}
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_3.log {1: 394.42, 3: 393.29}
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_2.log {1: 394.42, 3: 393.29, 2: 396.03}
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_5.log {1: 394.42, 3: 393.29, 2: 396.03, 5: 398.37}
fp32_b128/mxnet/resnet50/bz128/1n1g/rn50_b128_fp32_4.log {1: 394.42, 3: 393.29, 2: 396.03, 5: 398.37, 4: 400.5}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_1.log {1: 2614.52}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_3.log {1: 2614.52, 3: 2619.65}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_2.log {1: 2614.52, 3: 2619.65, 2: 2621.65}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_5.log {1: 2614.52, 3: 2619.65, 2: 2621.65, 5: 2615.47}
fp32_b128/mxnet/resnet50/bz128/1n8g/rn50_b128_fp32_4.log {1: 2614.52, 3: 2619.65, 2: 2621.65, 5: 2615.47, 4: 2633.38}
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_1.log {1: 768.16}
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_3.log {1: 768.16, 3: 770.27}
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_2.log {1: 768.16, 3: 770.27, 2: 761.0}
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_5.log {1: 768.16, 3: 770.27, 2: 761.0, 5: 770.34}
fp32_b128/mxnet/resnet50/bz128/1n2g/rn50_b128_fp32_4.log {1: 768.16, 3: 770.27, 2: 761.0, 5: 770.34, 4: 750.78}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_1.log {1: 9734.5}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_3.log {1: 9734.5, 3: 9750.65}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_2.log {1: 9734.5, 3: 9750.65, 2: 9755.77}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_5.log {1: 9734.5, 3: 9750.65, 2: 9755.77, 5: 9790.96}
fp32_b128/mxnet/resnet50/bz128/4n8g/rn50_b128_fp32_4.log {1: 9734.5, 3: 9750.65, 2: 9755.77, 5: 9790.96, 4: 9713.57}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_1.log {1: 5015.01}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_3.log {1: 5015.01, 3: 4976.45}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_2.log {1: 5015.01, 3: 4976.45, 2: 4981.01}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_5.log {1: 5015.01, 3: 4976.45, 2: 4981.01, 5: 4989.97}
fp32_b128/mxnet/resnet50/bz128/2n8g/rn50_b128_fp32_4.log {1: 5015.01, 3: 4976.45, 2: 4981.01, 5: 4989.97, 4: 4973.8}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_1.log {1: 1522.11}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_3.log {1: 1522.11, 3: 1522.69}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_2.log {1: 1522.11, 3: 1522.69, 2: 1508.74}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_5.log {1: 1522.11, 3: 1522.69, 2: 1508.74, 5: 1527.19}
fp32_b128/mxnet/resnet50/bz128/1n4g/rn50_b128_fp32_4.log {1: 1522.11, 3: 1522.69, 2: 1508.74, 5: 1527.19, 4: 1540.33}
{'rn50': {'1n1g': {'average_speed': 396.52,
                   'batch_size_per_device': 128,
                   'median_speed': 396.03,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 764.11,
                   'batch_size_per_device': 128,
                   'median_speed': 768.16,
                   'speedup': 1.94},
          '1n4g': {'average_speed': 1524.21,
                   'batch_size_per_device': 128,
                   'median_speed': 1522.69,
                   'speedup': 3.84},
          '1n8g': {'average_speed': 2620.93,
                   'batch_size_per_device': 128,
                   'median_speed': 2619.65,
                   'speedup': 6.61},
          '2n8g': {'average_speed': 4987.25,
                   'batch_size_per_device': 128,
                   'median_speed': 4981.01,
                   'speedup': 12.58},
          '4n8g': {'average_speed': 9749.09,
                   'batch_size_per_device': 128,
                   'median_speed': 9750.65,
                   'speedup': 24.62}}}
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

该小节提供针对 MXNet 框架的 ResNet50 v1.5 模型单机测试的性能结果和完整 log 日志。

### ResNet50 v1.5 FP32

#### batch_size = 128 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 396.03    | 1.00    |
| 1        | 2       | 764.11    | 1.94    |
| 1        | 4       | 1522.69   | 3.84    |
| 1        | 8       | 2619.65   | 6.61    |
| 2        | 16      | 4981.01   | 12.58   |
| 4        | 32      | 9750.65   | 24.62   |

### ResNet50 v1.5 FP16

#### batch_size = 128 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 751.54    | 1       |
| 1        | 2       | 1414.84   | 1.88    |
| 1        | 4       | 2651.61   | 3.53    |
| 1        | 8       | 3102.9    | 4.13    |
| 2        | 16      | 5814.59   | 7.74    |
| 4        | 32      | 11387.38  | 15.15   |


详细 Log 信息可下载：

- [mxnet_resnet50_fp32_b128.tar.gz](http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MxNet/resnet50/mxnet_resnet50_fp32_b128.tar.gz) 

- [mxnet_resnet50_fp16_b128.tar.gz](http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MxNet/resnet50/mxnet_resnet50_fp16_b128.tar.gz) 
