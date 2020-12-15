# MXNet InsightFace 测评

## 概述 Overview

本测试基于 [deepinsight](https://github.com/deepinsight/insightface/tree/863a7ea9ea0c0355d63c17e3c24e1373ed6bec55) 仓库中提供的基于MXNet框架的 [Partial-FC](https://github.com/deepinsight/insightface/tree/863a7ea9ea0c0355d63c17e3c24e1373ed6bec55/recognition/partial_fc) 实现，目的在于速度测评，同时根据测速结果给出单机～多机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试覆盖了FP32 精度下的单机1~8卡，后续将持续维护，增加更多方式的测评。



## 环境 Environment

### 系统

- #### 硬件

  - GPU：8 x Tesla V100-SXM2-16GB

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



## 快速开始 Quick Start

### 1. 前期准备

- #### 数据集

  数据集可以使用emore(MS1MV2-Arcface)、Glint360K等多种数据集。emore数据集准备及制作参考[ArcFace-readme](https://github.com/deepinsight/insightface/tree/863a7ea9ea0c0355d63c17e3c24e1373ed6bec55/recognition/ArcFace#model-training);Glint360K数据集的下载和准备过程参考[partial_fc-readme](https://github.com/deepinsight/insightface/tree/863a7ea9ea0c0355d63c17e3c24e1373ed6bec55/recognition/partial_fc#glint360k)，本次测试主要以emore数据集为主。

  

 - #### 环境安装
```shell
# 安装mxnet
python3 -m pip install gluonnlp==0.10.0 mxnet-cu102mkl==1.6.0.post0 -i https://mirror.baidu.com/pypi/simple
# 安装horovod（安装前，确保环境中已有nccl、openmpi）
HOROVOD_WITH_MXNET=1  HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL python3 -m pip install --no-cache-dir horovod==0.19.5
```


### 2. 额外准备

- 下载insightface仓库源码

  ```shell
  git clone https://github.com/deepinsight/insightface.git
  cd insightface && git checkout 863a7ea9ea0c0355d63c17e3c24e1373ed6bec55
  git submodule init
  git submodule update
  ```

- 设置数据集路径

  修改`insightface/recognition/partial_fc/mxnet/default.py`下相应数据集路径，如修改第63行：

  `config.rec = '/datasets/insightface/glint360k/train.rec'`以为glint360k_8GPU数据集设置本地路径。

- 注释模型报错相关代码

  由于是性能评测而非完整训练，我们不需要保存模型模型，可以注释掉`insightface/recognition/partial_fc/mxnet/callbacks.py`中第122行起保存模型相关的代码：

  ```shell
  # if self.rank == 0:
              #     mx.model.save_checkpoint(prefix=self.prefix + "_average",
              #                              epoch=0,
              #                              symbol=_sym,
              #                              arg_params=new_arg,
              #                              aux_params=new_aux)
              #     mx.model.save_checkpoint(prefix=self.prefix,
              #                              epoch=0,
              #                              symbol=_sym,
              #                              arg_params=arg,
              #                              aux_params=aux)
  ```

  


### 3. 运行测试

本次测试集群中有 4 台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里设置batch size为64，分别在单机1卡～单机8卡的情况下进行了多组训练。

#### 测试

在容器内下载本仓库源码：

````shell
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 `DLPerf/MXNet/InsightFace/PartialFC` 路径下的脚本和代码放至 `insightface/recognition/partial_fc/mxnet` 目录下，执行脚本

```shell
bash run_test.sh
```

针对1机1~8卡， 进行测试，并将 log 信息保存在logs目录下。



**默认测试的网络为resnet100，batch size=64 ，sample_ratio=1.0**，您也可以修改模型和相应的batch size如：

```shell
# 测试resnet100，batch size=96，sample_ratio=0.5
bash run_test.sh r100  96  0.5
# 测试resnet50，batch size=64，sample_ratio=1.0
bash run_test.sh r50   64  1.0
```




### 4. 数据处理

测试进行了多组训练（本测试中取 5 次），每次训练只进行120 iter，计算训练速度时去掉前 20 iter，只取后 100 iter 的数据，以降低抖动。最后将 5 次训练的速度取中位数得到最终速度，并最终以此数据计算加速比。

运行，即可得到针对不同配置测试结果 log 数据处理的结果： 

```shell
python extract_mxnet_logs.py --log_dir=./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64 --batch_size_per_device=64
```

结果打印如下

```shell
python extract_mxnet_logs.py --log_dir=./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64 --batch_size_per_device=64
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n8g/r100_b64_fp32_1.log {1: 1363.12}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n8g/r100_b64_fp32_5.log {1: 1363.12, 5: 1365.7}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n8g/r100_b64_fp32_3.log {1: 1363.12, 5: 1365.7, 3: 1350.14}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n8g/r100_b64_fp32_2.log {1: 1363.12, 5: 1365.7, 3: 1350.14, 2: 1351.64}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n8g/r100_b64_fp32_4.log {1: 1363.12, 5: 1365.7, 3: 1350.14, 2: 1351.64, 4: 1318.4}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n4g/r100_b64_fp32_1.log {1: 749.94}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n4g/r100_b64_fp32_5.log {1: 749.94, 5: 753.74}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n4g/r100_b64_fp32_3.log {1: 749.94, 5: 753.74, 3: 759.86}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n4g/r100_b64_fp32_2.log {1: 749.94, 5: 753.74, 3: 759.86, 2: 742.62}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n4g/r100_b64_fp32_4.log {1: 749.94, 5: 753.74, 3: 759.86, 2: 742.62, 4: 741.72}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n1g/r100_b64_fp32_1.log {1: 212.64}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n1g/r100_b64_fp32_5.log {1: 212.64, 5: 216.46}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n1g/r100_b64_fp32_3.log {1: 212.64, 5: 216.46, 3: 214.1}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n1g/r100_b64_fp32_2.log {1: 212.64, 5: 216.46, 3: 214.1, 2: 214.6}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n1g/r100_b64_fp32_4.log {1: 212.64, 5: 216.46, 3: 214.1, 2: 214.6, 4: 212.18}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n2g/r100_b64_fp32_1.log {1: 401.72}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n2g/r100_b64_fp32_5.log {1: 401.72, 5: 391.86}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n2g/r100_b64_fp32_3.log {1: 401.72, 5: 391.86, 3: 381.7}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n2g/r100_b64_fp32_2.log {1: 401.72, 5: 391.86, 3: 381.7, 2: 393.96}
./logs_1206_sample-ratio-1.0/mxnet/partial_fc/bz64/1n2g/r100_b64_fp32_4.log {1: 401.72, 5: 391.86, 3: 381.7, 2: 393.96, 4: 395.42}
{'r100': {'1n1g': {'average_speed': 214.0,
                   'batch_size_per_device': 64,
                   'median_speed': 214.1,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 392.93,
                   'batch_size_per_device': 64,
                   'median_speed': 393.96,
                   'speedup': 1.84},
          '1n4g': {'average_speed': 749.58,
                   'batch_size_per_device': 64,
                   'median_speed': 749.94,
                   'speedup': 3.5},
          '1n8g': {'average_speed': 1349.8,
                   'batch_size_per_device': 64,
                   'median_speed': 1351.64,
                   'speedup': 6.31}}}
Saving result to ./result/bz64_result.json
```



### 5. 计算规则

#### 5.1 测速脚本

- extract_mxnet_logs.py 根据官方在log中打印的速度，在120个iter中，排除前20iter，取后100个iter的速度做平均；

#### 5.2 均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度

#### 5.3 加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5



## 性能结果 Performance

- network:resnet100

- dataset:emore

- loss:arcface

### resnet100  FP32

#### batch size = 64 & sample ratio = 1.0

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 214.1     | 1.0     |
| 1        | 2       | 393.96    | 1.84    |
| 1        | 4       | 749.94    | 3.5     |
| 1        | 8       | 1351.64   | 6.31    |

#### Batch size = 104 & sample ratio = 1.0

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 193.16    | 1       |
| 1        | 2       | 329.26    | 1.7     |
| 1        | 4       | 599.12    | 3.1     |
| 1        | 8       | 812.7     | 4.21    |

#### batch size = 64 & sample ratio = 0.1

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 214.76    | 1       |
| 1        | 2       | 394.4     | 1.84    |
| 1        | 4       | 756.88    | 3.52    |
| 1        | 8       | 1345.24   | 6.26    |

#### Batch size = 104 &  sample ratio = 0.1

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 227.3     | 1       |
| 1        | 2       | 430.88    | 1.9     |
| 1        | 4       | 826.98    | 3.64    |
| 1        | 8       | 1532.58   | 6.74    |




### 日志下载

详细 Log 信息可点击下载：

- [arcface_fp32.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MxNet/insightface/partial_fc/logs.zip)


