# 基于 InsightFace 模型的深度学习框架性能评测报告

## 简介

人脸识别技术可以用来自动确定图像中人脸的身份，具有丰富的应用场景，譬如刷脸支付、识别医院黄牛号贩子，到新冠流行病学调查、逃犯追踪等。典型的模型有 InsightFace，它是一个开源的 2D&3D 人脸识别分析工具箱，原本基于 MXNet 实现，OneFlow 经过严格的网络、参数及配置对齐也实现了该模型。

本报告比较了 [oneflow_face](https://github.com/Oneflow-Inc/oneflow_face/tree/master) 和 [deepinsight](https://github.com/deepinsight) 两个仓库在 InsightFace 深度学习模型训练任务上的吞吐率。测试均采用相同的数据集、相同的硬件环境和算法，仅比较框架之间的速度差异。结果表明：OneFlow 在 InsightFace 模型上的性能以及分布式环境下的横向扩展能力优于其他框架。

## 目录
- [基于 InsightFace 模型的深度学习框架性能评测报告](#基于-insightface-模型的深度学习框架性能评测报告)
  - [简介](#简介)
  - [目录](#目录)
  - [数据来源](#数据来源)
  - [框架 & 模型](#框架--模型)
  - [评测配置](#评测配置)
    - [1. 网络对齐](#1-网络对齐)
    - [2. Batch Size](#2-batch-size)
    - [3. Num Classes](#3-num-classes)
  - [测试结果](#测试结果)
    - [Face Emore & R100 & FP32 Thoughput](#face-emore--r100--fp32-thoughput)
      - [Data Parallelism](#data-parallelism)
      - [Model Parallelism](#model-parallelism)
      - [Partial FC, sample_ratio = 0.1](#partial-fc-sample_ratio--01)
    - [Glint360k & R100 & FP32 Thoughputs](#glint360k--r100--fp32-thoughputs)
      - [Data Parallelism](#data-parallelism-1)
      - [Model Parallelism](#model-parallelism-1)
      - [Partial FC, sample_ratio = 0.1](#partial-fc-sample_ratio--01-1)
    - [Face Emore & Y1 & FP32 Thoughputs](#face-emore--y1--fp32-thoughputs)
      - [Data Parallelism](#data-parallelism-2)
      - [Model Parallelism](#model-parallelism-2)
    - [Max num_classses](#max-num_classses)
  - [测试结论](#测试结论)
## 数据来源

各个框架的性能评测复现流程、介绍、日志、数据以及英文版报告均可以在 DLPerf 仓库中查看： https://github.com/Oneflow-Inc/DLPerf

## 框架 & 模型

| 框架                                                          | 版本              | 模型来源                                                                                                            |
| ------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.3.4) | 0.3.4             | [oneflow_face](https://github.com/Oneflow-Inc/oneflow_face/tree/1705ae5b4cee6466f7abf75ba891984ec02b8ea3)                                                         |
| [deepinsight](https://github.com/deepinsight)                 | 2021-01-20 update | [deepinsight/insightface](https://github.com/deepinsight/insightface/tree/a9beb60971fb8115698859c35fdca721d6f75f5d) |

## 评测配置

### 1. 网络对齐

OneFlow 的实现与 MXNet 进行了严格对齐，主要包括：

|                    | [R100](https://github.com/deepinsight/insightface/blob/master/recognition/partial_fc/mxnet/default.py#L86)（ResNet100）+ face_emore | [R100](https://github.com/deepinsight/insightface/blob/master/recognition/partial_fc/mxnet/default.py#L86)（ResNet100）+ glint360k | [Y1](https://github.com/nlqq/insightface/blob/master/recognition/ArcFace/sample_config.py)（MobileFaceNet）+ face_emore |
| ------------------ | :---------------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| fc type            |                                                                  E                                                                  | FC                                                                                                                                 | GDC                                                                                                                     |
| optimizer          |                                                                 SGD                                                                 | SGD                                                                                                                                | SGD                                                                                                                     |
| kernel initializer |                                          random_normal_initializer(mean=0.0, stddev=0.01)                                           | random_normal_initializer(mean=0.0, stddev=0.01)                                                                                   | random_normal_initializer(mean=0.0, stddev=0.01)                                                                        |
| loss type          |                                                               arcface                                                               | cosface                                                                                                                            | arcface                                                                                                                 |
| regularizer        |                                                          Step Weight Decay                                                          | Step Weight Decay                                                                                                                  | Step Weight Decay                                                                                                       |
| lr_step            |                                                           [100000,160000]                                                           | [200000, 400000, 500000, 550000]                                                                                                   | [100000,160000,220000]                                                                                                  |
| scales             |                                                             [0.1, 0.01]                                                             | [0.1, 0.01, 0.001, 0.0001]                                                                                                         | [0.1, 0.01, 0.001]                                                                                                      |
| momentum           |                                                                 0.9                                                                 | 0.9                                                                                                                                | 0.9                                                                                                                     |
| weight decay       |                                                               0.0005                                                                | 0.0005                                                                                                                             | 0.0005                                                                                                                  |

### 2. Batch Size

在本报告中，batch size 表示的是深度学习训练过程中每个设备（GPU/卡）上的样例个数，简称 bsz（batch size per GPU）；在本测试中，将针对不同框架实现测试单机单卡、单机 4 卡、单机 8 卡的固定和最大 batch size。

### 3. Num Classes 

在本报告中，num classes 表示的是 InsightFace 支持的人脸类别数。在本测试中，将针对不同框架实现测试单机单卡、单机 8 卡的固定和最大 num classes。

## 测试结果


### Face Emore & R100 & FP32 Thoughput

#### Data Parallelism

**batch_size = 64**


| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 245.0             | 241.82          |
| 1        | 4                | 64                    | 923.23            | 655.56          |
| 1        | 8                | 64                    | 1836.8            | 650.8           |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_r100_fp32_b64_dp_en.png)


**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s（max bsz=96） | MXNet samples/s(max bsz=96) |
| -------- | ---------------- | ------------------------------- | --------------------------- |
| 1        | 1                | 250.71                          | 288.0                       |
| 1        | 4                | 972.8                           | 733.1                       |
| 1        | 8                | 1931.76                         | 749.42                      |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_r100_fp32_bmax_dp_en.png)

#### Model Parallelism

**batch_size = 64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 245.29            | 233.88          |
| 1        | 4                | 64                    | 938.83            | 651.44          |
| 1        | 8                | 64                    | 1854.15           | 756.96          |


![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_r100_fp32_b64_mp_en.png)

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bsz=115) | MXNet samples/s（max bsz=96） |
| -------- | ---------------- | ------------------------------ | ----------------------------- |
| 1        | 1                | 246.55                         | 242.2                         |
| 1        | 4                | 970.1                          | 724.26                        |
| 1        | 8                | 1921.87                        | 821.06                        |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_r100_fp32_bmax_mp_en.png)

#### Partial FC, sample_ratio = 0.1

**batch_size=64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 246.45            | 223.11          |
| 1        | 4                | 64                    | 948.96            | 799.19          |
| 1        | 8                | 64                    | 1872.81           | 1586.09         |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_r100_fp32_b64_pf_en.png)

**batch_size=max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bsz=120) | MXNet samples/s(max bsz=104) |
| -------- | ---------------- | ------------------------------ | ---------------------------- |
| 1        | 1                | 256.61                         | 232.56                       |
| 1        | 4                | 990.82                         | 852.4                        |
| 1        | 8                | 1962.76                        | 1644.42                      |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_r100_fp32_bmax_pf_en.png)


### Glint360k & R100 & FP32 Thoughputs

#### Data Parallelism

**batch_size = 64**


| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 230.22            | -               |
| 1        | 4                | 64                    | 847.71            | -               |
| 1        | 8                | 64                    | 1688.62           | -               |



**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bsz=85) | MXNet samples/s(max bsz=?) |
| -------- | ---------------- | ----------------------------- | -------------------------- |
| 1        | 1                | 229.94                        | -                          |
| 1        | 4                | 856.61                        | -                          |
| 1        | 8                | 1707.03                       | -                          |


#### Model Parallelism

**batch_size = 64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 230.33            | -               |
| 1        | 4                | 64                    | 912.24            | -               |
| 1        | 8                | 64                    | 1808.27           | -               |



**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bsz=100) | MXNet samples/s(max bsz=?) |
| -------- | ---------------- | ------------------------------ | -------------------------- |
| 1        | 1                | 231.86                         | -                          |
| 1        | 4                | 925.85                         | -                          |
| 1        | 8                | 1844.66                        | -                          |

> 注意： MXNet 数据并行和模型并行的数据缺失是因为 [insightface/recognition/ArcFace/](https://github.com/deepinsight/insightface/tree/b774d6a1b7c66655f07cc59ce98007ff2c19a11d) 路径下的脚本不支持 Glint360k 数据集。

#### Partial FC, sample_ratio = 0.1

**batch_size=64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 245.12            | 194.01          |
| 1        | 4                | 64                    | 945.44            | 730.29          |
| 1        | 8                | 64                    | 1858.57           | 1359.2          |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/glint360k_r100_fp32_b64_pf_en.png)

**batch_size=max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bsz=115) | MXNet samples/s(max bsz=96) |
| -------- | ---------------- | ------------------------------ | --------------------------- |
| 1        | 1                | 248.01                         | 192.18                      |
| 1        | 4                | 973.63                         | 811.34                      |
| 1        | 8                | 1933.88                        | 1493.51                     |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/glint360k_r100_fp32_bmax_pf_en.png)

### Face Emore & Y1 & FP32 Thoughputs

#### Data Parallelism

**batch_size = 256**


| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 256                   | 1961.52           | 786.94          |
| 1        | 4                | 256                   | 7354.49           | 1055.88         |
| 1        | 8                | 256                   | 14298.02          | 1031.1          |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_y1_fp32_b256_dp_en.png)

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bsz=350) | MXNet samples/s(max bsz=368) |
| -------- | ---------------- | ------------------------------ | ---------------------------- |
| 1        | 1                | 1969.66                        | 931.88                       |
| 1        | 4                | 7511.53                        | 1044.38                      |
| 1        | 8                | 14756.03                       | 1026.68                      |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_y1_fp32_bmax_dp_en.png)

#### Model Parallelism

**batch_size = 256**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 256                   | 1963.62           | 984.2           |
| 1        | 4                | 256                   | 7264.54           | 984.88          |
| 1        | 8                | 256                   | 14049.75          | 1030.58         |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_y1_fp32_b256_mp_en.png)

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bsz=400) | MXNet samples/s(max bsz=352) |
| -------- | ---------------- | ------------------------------ | ---------------------------- |
| 1        | 1                | 1969.65                        | 974.26                       |
| 1        | 4                | 7363.77                        | 1017.78                      |
| 1        | 8                | 14436.38                       | 1038.6                       |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_y1_fp32_bmax_mp_en.png)

### Max num_classses

| node_num | gpu_num_per_node | batch_size_per_device | FP16 | Model Parallel | Partial FC | OneFlow num_classes | MXNet   num_classes |
| -------- | ---------------- | --------------------- | ---- | -------------- | ---------- | ------------------- | ------------------- |
| 1        | 1                | 64                    | True | True           | True       | 2000000             | 1800000             |
| 1        | 8                | 64                    | True | True           | True       | 13500000            | 12000000            |



## 测试结论

以上这一系列的测试表明：

1. 随着 `batch_size_per_device` 的增大，MXNet 的吞吐率即使使用了 Partial FC 也很难有突破，而 OneFlow 则始终保持较为稳定的线性增长；
2. 相同条件下 OneFlow 支持更大规模的 `batch_size` 和 `num_classes` ，在单机 8 卡、 单卡 batch size 固定为 64 ，同样是使用 FP16、model_parallel、partial_fc 的情况下，OneFlow 所支持的 `num_classes` 数量是 MXNet 的 1.125 倍（1350 万 vs 1200 万）；

更多数据细节可移步 DLPerf 的 OneFlow 和 MXNet。