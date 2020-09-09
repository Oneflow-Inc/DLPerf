# MXNet BERT-Base 测评

## 概述 Overview

本测试基于 [gluon-nlp](https://github.com/dmlc/gluon-nlp) 仓库中提供的 MXNet框架的 [BERT-base](https://github.com/dmlc/gluon-nlp/tree/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert) 实现，在服务器上进行了1机1卡、1机8卡、2机16卡、4机32卡的结果复现及速度评测，得到吞吐率及加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试仅覆盖 FP32 精度，后续将持续维护，增加混合精度训练，XLA 等多种方式的测评。



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

  | Feature                         | BERT-Base MXNet |
| ------------------------------- | --------------- |
  | Horovod Multi-GPU               | Yes             |
| Horovod Multi-Node              | Yes             |
  | Automatic mixed precision (AMP) | No              |



## 快速开始 Quick Start

### 1. 前期准备

- #### 数据集

  数据集制作参考[gluon-nlp仓库提供的create_pretraining_data.py脚本](https://github.com/dmlc/gluon-nlp/blob/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert/create_pretraining_data.py)

  

- #### SSH 免密

  单机测试下无需配置，但测试2机、4机等多机情况下，则需要配置节点间的ssh免密登录，保证MXNet 的 mpi 分布式脚本运行时可以在单机上与其他节点互联。

  

 - #### 环境安装
```shell
# 安装mxnet
python3 -m pip install gluonnlp==0.10.0 mxnet-cu102mkl==1.6.0.post0 -i https://mirror.baidu.com/pypi/simple
# 安装horovod（安装前，确保环境中已有nccl、openmpi）
HOROVOD_WITH_MXNET=1  HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL python3 -m pip install --no-cache-dir horovod==0.19.5
```


### 2. 额外准备

- #### 下载gluon-nlp仓库源码

  ```shell
  git clone https://github.com/dmlc/gluon-nlp.git 
  git checkout 7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e
  ```

  > 注： 切换到这个分支才能跑，v0.9x分支不能跑，跑起来会卡住。(现象是GPU内存和计算都有被占用，同时可以看到有一个CPU线程被100%占用，但是发生了死锁）

- #### 注释参数

  - 注释掉 [`scripts/bert/data/pretrain.py`](https://github.com/dmlc/gluon-nlp/blob/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert/data/pretrain.py#L65) 的round_to参数。

    原因是round_to参数会报错：

    ```shell
    <stderr>:TypeError: __init__() got an unexpected keyword argument 'round_to'
    ```

  - 注释掉 [`/scripts/bert/run_pretraining.py`](https://github.com/dmlc/gluon-nlp/blob/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert/run_pretraining.py) 里跟eval_dir相关的逻辑:

    [line:95](https://github.com/dmlc/gluon-nlp/blob/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert/run_pretraining.py#L95)  data_eval允许为空。

    ```
    parser.add_argument('--data_eval', type=str, required=False,
    ```

    [line:443](https://github.com/dmlc/gluon-nlp/blob/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert/run_pretraining.py#L443)  不进行eval。

    ```python
    #    if data_eval:
    #        # eval data is always based on a fixed npz file.
    #        shuffle = False
    #        dataset_eval = get_pretrain_data_npz(data_eval, batch_size_eval,
    #                                             len(ctxs), shuffle, 1, vocab)
    #        evaluate(dataset_eval, model, ctxs, args.log_interval, args.dtype)
    ```

    原因是加上eval会卡住很久。

  - 训练200个iterations就结束：

    在 [`/scripts/bert/run_pretraining.py`](https://github.com/dmlc/gluon-nlp/blob/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert/run_pretraining.py) 中添加结束标记，用于在train step达到后就终止训练

    [line:290](https://github.com/dmlc/gluon-nlp/blob/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert/run_pretraining.py#L290)

    ```python
                 if step_num >= num_train_steps:
    +                end_of_batch = True
                     break
    ```

    

### 3. 运行测试

本次测试集群中有 4 台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有 8 张 V100 显卡， 每张显卡显存 16 G。

#### 测试

在容器内下载本仓库源码：

````shell
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 `DLPerf/MXNet/BERT/` 路径源码放至 `gluon-nlp/scripts/bert/` 下，执行脚本

```shell
bash run_test.sh
```

针对1机1卡、1机8卡、2机16卡、4机32卡， batch_size_per_device = **32** 进行测试，并将 log 信息保存在当前目录的`logs/mxnet/bert/bz32`对应分布式配置路径中。



默认对batch size32进行测试，您也可以指定其他大小的batch size，如：

```shell
# batch size = 48
bash run_test.sh 48
# batch size = 64
bash run_test.sh 64
```

#### GPU显存占用

- batch size=32: 10317MiB / 16160MiB

- batch size=48: 14021MiB / 16160MiB

- batch size=64: 14959MiB / 16160MiB

  

### 4. 数据处理

测试进行了多组训练（本测试中取 7 次），每次训练只进行200 iter(1个epoch)，计算训练速度时去掉前 100 iter，只取后 100 iter 的数据，以降低抖动。最后将 7 次训练的速度取中位数得到最终速度，并最终以此数据计算加速比。

运行，即可得到针对不同配置测试结果 log 数据处理的结果： 

```shell
python extract_mxnet_logs.py --log_dir=./logs/mxnet/bert/bz32 --batch_size_per_device=32
```

结果打印如下

```shell
./logs/mxnet/bert/bz32/4n8g/bert_base_b32_fp32_3.log {3: 3666.21}
./logs/mxnet/bert/bz32/4n8g/bert_base_b32_fp32_4.log {3: 3666.21, 4: 3698.92}
./logs/mxnet/bert/bz32/4n8g/bert_base_b32_fp32_6.log {3: 3666.21, 4: 3698.92, 6: 3658.23}
./logs/mxnet/bert/bz32/4n8g/bert_base_b32_fp32_1.log {3: 3666.21, 4: 3698.92, 6: 3658.23, 1: 3693.44}
./logs/mxnet/bert/bz32/4n8g/bert_base_b32_fp32_5.log {3: 3666.21, 4: 3698.92, 6: 3658.23, 1: 3693.44, 5: 3671.45}
./logs/mxnet/bert/bz32/4n8g/bert_base_b32_fp32_7.log {3: 3666.21, 4: 3698.92, 6: 3658.23, 1: 3693.44, 5: 3671.45, 7: 3673.42}
./logs/mxnet/bert/bz32/4n8g/bert_base_b32_fp32_2.log {3: 3666.21, 4: 3698.92, 6: 3658.23, 1: 3693.44, 5: 3671.45, 7: 3673.42, 2: 3668.26}
./logs/mxnet/bert/bz32/1n8g/bert_base_b32_fp32_3.log {3: 1036.56}
./logs/mxnet/bert/bz32/1n8g/bert_base_b32_fp32_4.log {3: 1036.56, 4: 1047.02}
./logs/mxnet/bert/bz32/1n8g/bert_base_b32_fp32_6.log {3: 1036.56, 4: 1047.02, 6: 1075.79}
./logs/mxnet/bert/bz32/1n8g/bert_base_b32_fp32_1.log {3: 1036.56, 4: 1047.02, 6: 1075.79, 1: 1061.91}
./logs/mxnet/bert/bz32/1n8g/bert_base_b32_fp32_5.log {3: 1036.56, 4: 1047.02, 6: 1075.79, 1: 1061.91, 5: 1058.6}
./logs/mxnet/bert/bz32/1n8g/bert_base_b32_fp32_7.log {3: 1036.56, 4: 1047.02, 6: 1075.79, 1: 1061.91, 5: 1058.6, 7: 1050.76}
./logs/mxnet/bert/bz32/1n8g/bert_base_b32_fp32_2.log {3: 1036.56, 4: 1047.02, 6: 1075.79, 1: 1061.91, 5: 1058.6, 7: 1050.76, 2: 1075.74}
./logs/mxnet/bert/bz32/1n4g/bert_base_b32_fp32_3.log {3: 549.03}
./logs/mxnet/bert/bz32/1n4g/bert_base_b32_fp32_4.log {3: 549.03, 4: 538.91}
./logs/mxnet/bert/bz32/1n4g/bert_base_b32_fp32_6.log {3: 549.03, 4: 538.91, 6: 536.96}
./logs/mxnet/bert/bz32/1n4g/bert_base_b32_fp32_1.log {3: 549.03, 4: 538.91, 6: 536.96, 1: 537.05}
./logs/mxnet/bert/bz32/1n4g/bert_base_b32_fp32_5.log {3: 549.03, 4: 538.91, 6: 536.96, 1: 537.05, 5: 533.45}
./logs/mxnet/bert/bz32/1n4g/bert_base_b32_fp32_7.log {3: 549.03, 4: 538.91, 6: 536.96, 1: 537.05, 5: 533.45, 7: 549.75}
./logs/mxnet/bert/bz32/1n4g/bert_base_b32_fp32_2.log {3: 549.03, 4: 538.91, 6: 536.96, 1: 537.05, 5: 533.45, 7: 549.75, 2: 538.5}
./logs/mxnet/bert/bz32/1n1g/bert_base_b32_fp32_3.log {3: 149.53}
./logs/mxnet/bert/bz32/1n1g/bert_base_b32_fp32_4.log {3: 149.53, 4: 149.88}
./logs/mxnet/bert/bz32/1n1g/bert_base_b32_fp32_6.log {3: 149.53, 4: 149.88, 6: 150.04}
./logs/mxnet/bert/bz32/1n1g/bert_base_b32_fp32_1.log {3: 149.53, 4: 149.88, 6: 150.04, 1: 150.11}
./logs/mxnet/bert/bz32/1n1g/bert_base_b32_fp32_5.log {3: 149.53, 4: 149.88, 6: 150.04, 1: 150.11, 5: 150.12}
./logs/mxnet/bert/bz32/1n1g/bert_base_b32_fp32_7.log {3: 149.53, 4: 149.88, 6: 150.04, 1: 150.11, 5: 150.12, 7: 150.9}
./logs/mxnet/bert/bz32/1n1g/bert_base_b32_fp32_2.log {3: 149.53, 4: 149.88, 6: 150.04, 1: 150.11, 5: 150.12, 7: 150.9, 2: 150.14}
./logs/mxnet/bert/bz32/1n2g/bert_base_b32_fp32_3.log {3: 273.02}
./logs/mxnet/bert/bz32/1n2g/bert_base_b32_fp32_4.log {3: 273.02, 4: 270.28}
./logs/mxnet/bert/bz32/1n2g/bert_base_b32_fp32_6.log {3: 273.02, 4: 270.28, 6: 272.09}
./logs/mxnet/bert/bz32/1n2g/bert_base_b32_fp32_1.log {3: 273.02, 4: 270.28, 6: 272.09, 1: 267.29}
./logs/mxnet/bert/bz32/1n2g/bert_base_b32_fp32_5.log {3: 273.02, 4: 270.28, 6: 272.09, 1: 267.29, 5: 269.99}
./logs/mxnet/bert/bz32/1n2g/bert_base_b32_fp32_7.log {3: 273.02, 4: 270.28, 6: 272.09, 1: 267.29, 5: 269.99, 7: 268.86}
./logs/mxnet/bert/bz32/1n2g/bert_base_b32_fp32_2.log {3: 273.02, 4: 270.28, 6: 272.09, 1: 267.29, 5: 269.99, 7: 268.86, 2: 271.05}
./logs/mxnet/bert/bz32/2n8g/bert_base_b32_fp32_3.log {3: 1854.88}
./logs/mxnet/bert/bz32/2n8g/bert_base_b32_fp32_4.log {3: 1854.88, 4: 1844.1}
./logs/mxnet/bert/bz32/2n8g/bert_base_b32_fp32_6.log {3: 1854.88, 4: 1844.1, 6: 1835.11}
./logs/mxnet/bert/bz32/2n8g/bert_base_b32_fp32_5.log {3: 1854.88, 4: 1844.1, 6: 1835.11, 5: 1847.2}
./logs/mxnet/bert/bz32/2n8g/bert_base_b32_fp32_7.log {3: 1854.88, 4: 1844.1, 6: 1835.11, 5: 1847.2, 7: 1841.41}
./logs/mxnet/bert/bz32/2n8g/bert_base_b32_fp32_2.log {3: 1854.88, 4: 1844.1, 6: 1835.11, 5: 1847.2, 7: 1841.41, 2: 1848.74}
{'bert-base': {'1n1g': {'average_speed': 150.1,
                        'batch_size_per_device': 32,
                        'median_speed': 150.11,
                        'speedup': 1.0},
               '1n2g': {'average_speed': 270.37,
                        'batch_size_per_device': 32,
                        'median_speed': 270.28,
                        'speedup': 1.8},
               '1n4g': {'average_speed': 540.52,
                        'batch_size_per_device': 32,
                        'median_speed': 538.5,
                        'speedup': 3.59},
               '1n8g': {'average_speed': 1058.05,
                        'batch_size_per_device': 32,
                        'median_speed': 1058.6,
                        'speedup': 7.05},
               '2n8g': {'average_speed': 1845.24,
                        'batch_size_per_device': 32,
                        'median_speed': 1845.65,
                        'speedup': 12.3},
               '4n8g': {'average_speed': 3675.7,
                        'batch_size_per_device': 32,
                        'median_speed': 3671.45,
```



### 5. 计算规则

#### 5.1 测速脚本

- extract_mxnet_logs.py 根据官方在log中打印的速度，在200个iter中，排除前100iter，取后100个iter的速度做平均；



#### 5.2 均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行7次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度

#### 5.3 加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5



## 性能结果 Performance

该小节提供针对 MXNet 框架的BERT-base 模型单机测试的性能结果和完整 log 日志。

### FP32 & W/O XLA

- ### BERT-base batch_size = 64

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 156.76    | 1.00    |
| 1        | 2       | 295.42    | 1.88    |
| 1        | 4       | 587.71    | 3.75    |
| 1        | 8       | 1153.08   | 7.36    |
| 2        | 16      | 2172.62   | 13.86   |
| 4        | 32      | 4340.89   | 27.69   |



- ### BERT-base batch_size = 48

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 153.75    | 1.00    |
| 1        | 2       | 287.77    | 1.87    |
| 1        | 4       | 572.64    | 3.72    |
| 1        | 8       | 1127.41   | 7.33    |
| 2        | 16      | 2067.72   | 13.45   |
| 4        | 32      | 4105.29   | 26.7    |



- ### BERT-base batch_size = 32

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 150.11    | 1.00    |
| 1        | 2       | 270.28    | 1.80    |
| 1        | 4       | 538.5     | 3.59    |
| 1        | 8       | 1058.6    | 7.05    |
| 2        | 16      | 1845.65   | 12.30   |
| 4        | 32      | 3671.45   | 24.46   |

详细 Log 信息可下载：[mxnet_bert_base_logs.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MxNet/bert/logs.zip)
