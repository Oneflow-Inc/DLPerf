# OneFlow InsightFace(r50) 测评


## 概述 Overview

本测试基于 OneFlow框架，对 [oneflow_face](https://github.com/Oneflow-Inc/oneflow_face/tree/dev_test_dlperf_rn50)  仓库中的 InsightFace 网络进行了从单机单卡到多机多卡的评测，评判框架的训练速度以及在分布式训练情况下的横向拓展能力。

本次测评使用的backbone为r50网络，在数据并行、模型并行的情况下分别进行了测试。




## 环境 Environment


### 系统


- #### 硬件


  - GPU：Tesla V100-SXM2-16GB x 8


- #### 软件


  - 驱动：NVIDIA 440.33.01


  - 系统：[ Ubuntu 16.04](http://releases.ubuntu.com/16.04/)

  - CUDA：10.2


  - cuDNN：7.6.5


- OneFlow：0.34
- Python： 3.7



  #### Feature support matrix

| Feature                         | ResNet50 v1.5 PyTorch |
| ------------------------------- | --------------------- |
| Multi-gpu training              | Yes                   |
| Automatic mixed precision (AMP) | Yes                   |
| Model Parallelism               | Yes                   |
| Partial FC                      | Yes                   |




## 快速开始 Quick Start


### 1. 前期准备


- #### 数据集

准备 Face Emore 和 Glint360k 的 OFReocord 数据集，可以选择根据 [加载与准备 OFRecord 数据集](https://docs.oneflow.org/extended_topics/how_to_make_ofdataset.html)文档中 Python 脚本生成所有数据的完整 OFRecord + Spark Shuffle + Spark Partition 的方式，也可以选择只使用 Python 脚本生成多块 OFRecord 的方式，用以进行 InsightFace 的测试。

可以参考 deepinsight [Training Data](https://github.com/deepinsight/insightface#training-data) 小节，下载 [Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) 中的 [MS1M-ArcFace](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ) 数据集或者 [Glint360k](https://pan.baidu.com/share/init?surl=GsYqTTt7_Dn8BfxxsLFN0w) 数据集。

具体制作方法请参考 [InsightFace 在 OneFlow 中的实现](https://github.com/Oneflow-Inc/oneflow_face/blob/master/README_CH.md#insightface-%E5%9C%A8-oneflow-%E4%B8%AD%E7%9A%84%E5%AE%9E%E7%8E%B0)中的[准备数据集](https://github.com/Oneflow-Inc/oneflow_face/blob/master/README_CH.md#%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE%E9%9B%86)部分。

更多关于 OneFlow OFRecord 数据集的信息，请参考 [加载与准备 OFRecord 数据集](https://docs.oneflow.org/extended_topics/how_to_make_ofdataset.html) 和 [将图片文件制作为 OFRecord 数据集](https://docs.oneflow.org/extended_topics/how_to_convert_image_to_ofrecord.html)。






### 2. 运行测试


本次测试集群中有 4 台节点：


- NODE1=10.11.0.2

- NODE2=10.11.0.3

- NODE3=10.11.0.4

- NODE4=10.11.0.5


每个节点有 8 张 V100 显卡， 每张显卡显存 16 G。


- #### 单机测试


在节点 1 的容器内下载 oneflow_face 源码和本仓库源码：


````
git clone https://github.com/Oneflow-Inc/oneflow_face.git
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 DLPerf/OneFlow/Recognition/InsightFace/r50/ 路径下的scripts文件夹复制到 oneflow_face 路径下。

sample_config.py中配置好数据集及其他默认参数后，执行以下命令在单机单卡、4 卡、8 卡情况下进行测试：

```
bash scripts/run_single_node.sh
```
默认batch size为128，使用r50的backbone网络，emore数据集，arcface的loss。

也可以修改脚本`run_single_node.sh` 中的参数以测试其他选项，如设置`model_parallel=${9:-1}`以开启全连接层的模型并行。

- #### 多机测试
其中运行 `bash scripts/run_two_node.sh` 以进行2机测试，运行`bash scripts/run_multi_node.sh` 以进行4机测试


### 3. 数据处理

测试进行了多组训练（本测试中取 5 次），每次训练过程取第 1 个 epoch 的前 150 iter，计算训练速度时取后 100 iter 的数据，以降低抖动。最后将 5 次训练的结果取中位数得到最终速度，并最终以此数据计算加速比。


运行 DLPerf/OneFlow/Recognition/InsightFace/rn50/extract_pytorch_logs_time.py，即可得到针对不同配置测试结果 log 数据处理的结果： 


```
python extract_oneflow_logs_time.py -ld 20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/
```

结果打印如下


```
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/4n8g/r50_b128_fp32_1.log {1: 12281.66}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/4n8g/r50_b128_fp32_4.log {1: 12281.66, 4: 12320.24}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/4n8g/r50_b128_fp32_2.log {1: 12281.66, 4: 12320.24, 2: 12397.28}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/4n8g/r50_b128_fp32_3.log {1: 12281.66, 4: 12320.24, 2: 12397.28, 3: 12373.61}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/4n8g/r50_b128_fp32_5.log {1: 12281.66, 4: 12320.24, 2: 12397.28, 3: 12373.61, 5: 12285.6}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n8g/r50_b128_fp32_1.log {1: 3248.24}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n8g/r50_b128_fp32_4.log {1: 3248.24, 4: 3285.21}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n8g/r50_b128_fp32_2.log {1: 3248.24, 4: 3285.21, 2: 3286.14}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n8g/r50_b128_fp32_3.log {1: 3248.24, 4: 3285.21, 2: 3286.14, 3: 3278.55}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n8g/r50_b128_fp32_5.log {1: 3248.24, 4: 3285.21, 2: 3286.14, 3: 3278.55, 5: 3276.42}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n4g/r50_b128_fp32_1.log {1: 1649.55}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n4g/r50_b128_fp32_4.log {1: 1649.55, 4: 1653.18}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n4g/r50_b128_fp32_2.log {1: 1649.55, 4: 1653.18, 2: 1652.16}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n4g/r50_b128_fp32_3.log {1: 1649.55, 4: 1653.18, 2: 1652.16, 3: 1654.45}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n4g/r50_b128_fp32_5.log {1: 1649.55, 4: 1653.18, 2: 1652.16, 3: 1654.45, 5: 1651.67}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n1g/r50_b128_fp32_1.log {1: 425.11}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n1g/r50_b128_fp32_4.log {1: 425.11, 4: 424.65}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n1g/r50_b128_fp32_2.log {1: 425.11, 4: 424.65, 2: 424.75}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n1g/r50_b128_fp32_3.log {1: 425.11, 4: 424.65, 2: 424.75, 3: 424.84}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/1n1g/r50_b128_fp32_5.log {1: 425.11, 4: 424.65, 2: 424.75, 3: 424.84, 5: 424.35}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/2n8g/r50_b128_fp32_1.log {1: 6330.97}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/2n8g/r50_b128_fp32_4.log {1: 6330.97, 4: 6343.74}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/2n8g/r50_b128_fp32_2.log {1: 6330.97, 4: 6343.74, 2: 6340.23}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/2n8g/r50_b128_fp32_3.log {1: 6330.97, 4: 6343.74, 2: 6340.23, 3: 6354.92}
20210319_r50_fp32_b128_oneflow_model_parallel_1_partial_fc_0/2n8g/r50_b128_fp32_5.log {1: 6330.97, 4: 6343.74, 2: 6340.23, 3: 6354.92, 5: 6384.45}
{'r50': {'1n1g': {'average_speed': 424.74,
                  'batch_size_per_device': 128,
                  'median_speed': 424.75,
                  'speedup': 1.0},
         '1n4g': {'average_speed': 1652.2,
                  'batch_size_per_device': 128,
                  'median_speed': 1652.16,
                  'speedup': 3.89},
         '1n8g': {'average_speed': 3274.91,
                  'batch_size_per_device': 128,
                  'median_speed': 3278.55,
                  'speedup': 7.72},
         '2n8g': {'average_speed': 6350.86,
                  'batch_size_per_device': 128,
                  'median_speed': 6343.74,
                  'speedup': 14.94},
         '4n8g': {'average_speed': 12331.68,
                  'batch_size_per_device': 128,
                  'median_speed': 12320.24,
                  'speedup': 29.01}}}
Saving result to ./result/_result.json
```



## 性能结果 Performance


该小节提供针对 OneFlow 框架的 InsightFace 模型单机测试的性能结果和完整 log 日志。

### Face Emore & R50 & FP32 

#### Data Parallelism

**batch_size = 128**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 128                | 424.57 | 1.00    |
| 1        | 4                | 128                | 1635.63 | 3.85 |
| 1        | 8                | 128                | 3266.08 | 7.69 |
| 2        | 8                | 128                | 5827.13 | 13.72 |
| 4        | 8                | 128                | 11383.94 | 26.81 |

### Face Emore & R50 & FP32 

#### Model Parallelism

**batch_size = 128**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 128                   | 424.75    | 1.00    |
| 1        | 4                | 128                   | 1652.16   | 3.89    |
| 1        | 8                | 128                   | 3278.55   | 7.72    |
| 2        | 8                | 128                   | 6343.74   | 14.94   |
| 4        | 8                | 128                   | 12320.24  | 29.01   |

详细 Log 信息可下载：[logs-20210319.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/InsightFace/r50/logs-20210319.zip)