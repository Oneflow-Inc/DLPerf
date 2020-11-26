# MXNet InsightFace 测评

## 概述 Overview

本测试基于 [deepinsight](https://github.com/deepinsight/insightface/tree/863a7ea9ea0c0355d63c17e3c24e1373ed6bec55) 仓库中提供的基于MXNet框架的 [insightface](https://github.com/deepinsight/insightface/tree/863a7ea9ea0c0355d63c17e3c24e1373ed6bec55/recognition/ArcFace) 实现，进行了1机1～8卡的结果复现及速度评测，得到吞吐率及加速比，评判框架在真实训练环境下的速度及横向拓展能力。

目前，该测试覆盖了FP32 精度，后续将持续维护，增加更多方式的测评。



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



## 快速开始 Quick Start

### 1. 前期准备

- #### 数据集

  数据集准备及制作参考[deepinsight官方readme中Model Training部分的说明](https://github.com/deepinsight/insightface/tree/863a7ea9ea0c0355d63c17e3c24e1373ed6bec55/recognition/ArcFace#model-training)

  

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

- 设置配置文件：

  ```
  cp sample_config.py config.py
  vim config.py # edit dataset path etc..
  ```

- 修改config.py[第23行](https://github.com/deepinsight/insightface/blob/863a7ea9ea0c0355d63c17e3c24e1373ed6bec55/recognition/ArcFace/sample_config.py#L23)，设置config.max_steps = 120以使得程序训练120iter自动退出


### 3. 运行测试

本次测试集群中有 4 台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有 8 张 V100 显卡， 每张显卡显存 16 G。目前，由于官方尚未提供多机训练的代码，我们的测试仅在一台节点上进行。

#### 测试

在容器内下载本仓库源码：

````shell
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 `DLPerf/MXNet/InsightFace/` 路径下的脚本和代码放至 `insightface/recognition/ArcFace` 目录下，执行脚本

```shell
bash run_test.sh
```

针对1机1卡、1机1卡、1机4卡、1机8卡， batch_size_per_device = **64** 进行测试，并将 log 信息保存在当前目录下。

**默认测试的网络为resnet100，batch size=64** ，您也可以修改模型和相应的batch size如：

```shell
# 测试resnet100，batch size=96
bash   run_test.sh   r100   96
# 测试mobilefacenet，batch size=128
bash   run_test.sh y1   128
```




### 4. 数据处理

测试进行了多组训练（本测试中取 5 次），每次训练只进行120 iter(1个epoch)，计算训练速度时去掉前 20 iter，只取后 100 iter 的数据，以降低抖动。最后将 5 次训练的速度取中位数得到最终速度，并最终以此数据计算加速比。

运行，即可得到针对不同配置测试结果 log 数据处理的结果： 

```shell
python extract_mxnet_logs.py --log_dir=./logs/insightface/arcface/bz64 --batch_size_per_device=64
```

结果打印如下

```shell
./logs/insightface/arcface/bz64/1n8g/r100_b64_fp32_1.log {1: 765.7}
./logs/insightface/arcface/bz64/1n8g/r100_b64_fp32_5.log {1: 765.7, 5: 756.96}
./logs/insightface/arcface/bz64/1n8g/r100_b64_fp32_3.log {1: 765.7, 5: 756.96, 3: 739.02}
./logs/insightface/arcface/bz64/1n8g/r100_b64_fp32_2.log {1: 765.7, 5: 756.96, 3: 739.02, 2: 749.12}
./logs/insightface/arcface/bz64/1n8g/r100_b64_fp32_4.log {1: 765.7, 5: 756.96, 3: 739.02, 2: 749.12, 4: 763.2}
./logs/insightface/arcface/bz64/1n4g/r100_b64_fp32_1.log {1: 639.06}
./logs/insightface/arcface/bz64/1n4g/r100_b64_fp32_5.log {1: 639.06, 5: 651.44}
./logs/insightface/arcface/bz64/1n4g/r100_b64_fp32_3.log {1: 639.06, 5: 651.44, 3: 661.08}
./logs/insightface/arcface/bz64/1n4g/r100_b64_fp32_2.log {1: 639.06, 5: 651.44, 3: 661.08, 2: 658.46}
./logs/insightface/arcface/bz64/1n4g/r100_b64_fp32_4.log {1: 639.06, 5: 651.44, 3: 661.08, 2: 658.46, 4: 649.76}
./logs/insightface/arcface/bz64/1n1g/r100_b64_fp32_1.log {1: 233.94}
./logs/insightface/arcface/bz64/1n1g/r100_b64_fp32_5.log {1: 233.94, 5: 238.78}
./logs/insightface/arcface/bz64/1n1g/r100_b64_fp32_3.log {1: 233.94, 5: 238.78, 3: 233.52}
./logs/insightface/arcface/bz64/1n1g/r100_b64_fp32_2.log {1: 233.94, 5: 238.78, 3: 233.52, 2: 231.46}
./logs/insightface/arcface/bz64/1n1g/r100_b64_fp32_4.log {1: 233.94, 5: 238.78, 3: 233.52, 2: 231.46, 4: 233.88}
./logs/insightface/arcface/bz64/1n2g/r100_b64_fp32_1.log {1: 448.18}
./logs/insightface/arcface/bz64/1n2g/r100_b64_fp32_5.log {1: 448.18, 5: 449.54}
./logs/insightface/arcface/bz64/1n2g/r100_b64_fp32_3.log {1: 448.18, 5: 449.54, 3: 450.4}
./logs/insightface/arcface/bz64/1n2g/r100_b64_fp32_2.log {1: 448.18, 5: 449.54, 3: 450.4, 2: 451.18}
./logs/insightface/arcface/bz64/1n2g/r100_b64_fp32_4.log {1: 448.18, 5: 449.54, 3: 450.4, 2: 451.18, 4: 451.56}
{'r100': {'1n1g': {'average_speed': 234.32,
                   'batch_size_per_device': 64,
                   'median_speed': 233.88,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 450.17,
                   'batch_size_per_device': 64,
                   'median_speed': 450.4,
                   'speedup': 1.93},
          '1n4g': {'average_speed': 651.96,
                   'batch_size_per_device': 64,
                   'median_speed': 651.44,
                   'speedup': 2.79},
          '1n8g': {'average_speed': 754.8,
                   'batch_size_per_device': 64,
                   'median_speed': 756.96,
                   'speedup': 3.24}}}
Saving result to ./result/bz64_result.json
```



### 5. 计算规则

#### 5.1 测速脚本

- extract_mxnet_logs.py 根据官方在log中打印的速度，在120个iter中，排除前100iter，取后100个iter的速度做平均；

#### 5.2 均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度

#### 5.3 加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5



## 性能结果 Performance

该小节提供针对 MXNet 框架的BERT-base 模型单机测试的性能结果和完整 log 日志。

### ArcFace(resnet100) FP32

#### Batch size = 64 & Without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 233.88    | 1       |
| 1        | 2       | 450.4     | 1.93    |
| 1        | 4       | 651.44    | 2.79    |
| 1        | 8       | 756.96    | 3.24    |

#### Batch size = 96 & Without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 242.2     | 1       |
| 1        | 2       | 466.02    | 1.92    |
| 1        | 4       | 724.26    | 2.99    |
| 1        | 8       | 821.06    | 3.39    |

### ArcFace(mobilefacenet) FP32

#### Batch size = 128 & Without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 974.32    | 1       |
| 1        | 2       | 856.44    | 0.88    |
| 1        | 4       | 934.88    | 0.96    |
| 1        | 8       | 1000.76   | 1.03    |

#### Batch size = 256 & Without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 984.2     | 1       |
| 1        | 2       | 953.02    | 0.97    |
| 1        | 4       | 984.88    | 1.0     |
| 1        | 8       | 1030.58   | 1.05    |

#### Batch size = 352 & Without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 974.26 | 1       |
| 1        | 2       | 955.58 | 0.98 |
| 1        | 4       | 1017.78 | 1.04 |
| 1        | 8       | 1038.6 | 1.07  |

详细 Log 信息可下载：

- [arcface_fp32.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MxNet/insightface/arcface_fp32.zip)

  


