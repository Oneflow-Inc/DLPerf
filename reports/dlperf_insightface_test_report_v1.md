# InsightFace Deeplearning Framework Tests Report

## Introduction

Face recognization could automatically determine the identity of the face in the image and has got rich application scenarios, such as facial payment, identification of traffickers in hospital scalpers, epidemiological investigation of the new crown, fugitive tracking, etc. Typically, InsightFace, an open-source 2D&3D deep face analysis toolbox, mainly based on MXNet before, now, OneFlow has implemented it after strict alignment of network, parameters, and configuration. 

The report compares throughputs of InsightFace model between repository of  [oneflow_face](https://github.com/Oneflow-Inc/oneflow_face/tree/master) and [deepinsight](https://github.com/deepinsight). With the same datasets and hardware environment and algorithm, Only speed performances have been compared. In conclusion, OneFlow is better in performance of training InsightFace and distribution scalability. 

## Content
- [InsightFace Deeplearning Framework Tests Report](#insightface-deeplearning-framework-tests-report)
  - [Introduction](#introduction)
  - [Content](#content)
  - [Data](#data)
  - [Frameworks & MOdels](#frameworks--models)
  - [Configration](#configration)
    - [1. Network Alignment](#1-network-alignment)
    - [2. Batch Size](#2-batch-size)
    - [3. Num Classes](#3-num-classes)
  - [Results](#results)
    - [Face Emore & R100 & FP32 Thoughput](#face-emore--r100--fp32-thoughput)
      - [Data Parallelism](#data-parallelism)
      - [Model Parallelism](#model-parallelism)
      - [Partial FC, sample_ratio=0.1](#partial-fc-sample_ratio01)
    - [Glint360k & R100 & FP32 Thoughputs](#glint360k--r100--fp32-thoughputs)
      - [Data Parallelism](#data-parallelism-1)
      - [Model Parallelism](#model-parallelism-1)
      - [Partial FC, sample_ratio=0.1](#partial-fc-sample_ratio01-1)
    - [Face Emore & Y1 & FP32 Thoughputs](#face-emore--y1--fp32-thoughputs)
      - [Data Parallelism](#data-parallelism-2)
      - [Model Parallelism](#model-parallelism-2)
    - [Max num_classses](#max-num_classses)
  - [Conclusion](#conclusion)
## Data

Reproduction procedures, introductions, logs, data, and English reports could be fetched in DLPerf repository: https://github.com/Oneflow-Inc/DLPerf


## Frameworks & MOdels

| 框架                                                         | 版本              | 模型来源                                                     |
| ------------------------------------------------------------ | ----------------- | ------------------------------------------------------------ |
| [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.3.4) | 0.3.4             | [oneflow_face]()                                             |
| [deepinsight](https://github.com/deepinsight)                | 2021-01-20 update | [deepinsight/insightface](https://github.com/deepinsight/insightface/tree/a9beb60971fb8115698859c35fdca721d6f75f5d) |

## Configration

### 1. Network Alignment 

rigorous alignment has been completed between OneFlow and MxNet, including:


|                    | [R100](https://github.com/deepinsight/insightface/blob/master/recognition/partial_fc/mxnet/default.py#L86)（ResNet100）+ face_emore | [R100](https://github.com/deepinsight/insightface/blob/master/recognition/partial_fc/mxnet/default.py#L86)（ResNet100）+ glint360k | [Y1](https://github.com/nlqq/insightface/blob/master/recognition/ArcFace/sample_config.py)（MobileFaceNet）+ face_emore |
| ------------------ | :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| fc type            |                              E                               | FC                                                           | GDC                                                          |
| optimizer          |                             SGD                              | SGD                                                          | SGD                                                          |
| kernel initializer |       random_normal_initializer(mean=0.0, stddev=0.01)       | random_normal_initializer(mean=0.0, stddev=0.01)             | random_normal_initializer(mean=0.0, stddev=0.01)             |
| loss type          |                           arcface                            | cosface                                                      | arcface                                                      |
| regularizer        |                      Step Weight Decay                       | Step Weight Decay                                            | Step Weight Decay                                            |
| lr_step            |                       [100000,160000]                        | [200000, 400000, 500000, 550000]                             | [100000,160000,220000]                                       |
| scales             |                         [0.1, 0.01]                          | [0.1, 0.01, 0.001, 0.0001]                                   | [0.1, 0.01, 0.001]                                           |
| momentum           |                             0.9                              | 0.9                                                          | 0.9                                                          |
| weight decay       |                            0.0005                            | 0.0005                                                       | 0.0005                                                       |

### 2. Batch Size

In this report, batch size means the number of samples on each device(GPU), bsz (batch size per GPU) in short. In the tests, it will give the static value or maximum of batch size with different numbers of GPU tests in different frameworks.

### 3. Num Classes 

In this report, num classes mean the number of face categories. In the tests, it will give the static value or maximum of num classes with different numbers of GPU tests in different frameworks.

## Results

### Face Emore & R100 & FP32 Thoughput

#### Data Parallelism

**batch_size = 64**


| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 245.0             |                 |
| 1        | 4                | 64                    | 923.23            |                 |
| 1        | 8                | 64                    | 1836.8            |                 |




**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s（max bzs=96） | MXNet samples/s(max bzs=96) |
| -------- | ---------------- | ------------------------------- | --------------------------- |
| 1        | 1                | 252.76                          |                             |
| 1        | 4                | 969.27                          |                             |
| 1        | 8                | 1925.6                          |                             |



#### Model Parallelism

**batch_size = 64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 245.29            | 223.88          |
| 1        | 4                | 64                    | 938.83            | 651.44          |
| 1        | 8                | 64                    | 1854.15           | 756.96          |

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bzs=115) | MXNet samples/s（max bzs=96） |
| -------- | ---------------- | ------------------------------ | ----------------------------- |
| 1        | 1                | 245.92                         | 242.2                         |
| 1        | 4                | 968.72                         | 724.26                        |
| 1        | 8                | 1925.59                        | 821.06                        |

#### Partial FC, sample_ratio=0.1

**batch_size=64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 247.97            |                 |
| 1        | 4                | 64                    | 946.54            |                 |
| 1        | 8                | 64                    | 1864.31           |                 |

**batch_size=max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bzs=120) | MXNet samples/s(max bzs=?) |
| -------- | ---------------- | ------------------------------ | -------------------------- |
| 1        | 1                | 256.61                         |                            |
| 1        | 4                | 990.82                         |                            |
| 1        | 8                | 1962.76                        |                            |



### Glint360k & R100 & FP32 Thoughputs

#### Data Parallelism

**batch_size = 64**


| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 230.22            |                 |
| 1        | 4                | 64                    | 847.71            |                 |
| 1        | 8                | 64                    | 1688.62           |                 |

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bzs=85) | MXNet samples/s(max bzs=?) |
| -------- | ---------------- | ----------------------------- | -------------------------- |
| 1        | 1                | 229.94                        |                            |
| 1        | 4                | 856.61                        |                            |
| 1        | 8                | 1707.03                       |                            |

#### Model Parallelism

**batch_size = 64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 230.33            |                 |
| 1        | 4                | 64                    | 912.24            |                 |
| 1        | 8                | 64                    | 1808.27           |                 |

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bzs=100) | MXNet samples/s(max bzs=?) |
| -------- | ---------------- | ------------------------------ | -------------------------- |
| 1        | 1                | 231.86                         |                            |
| 1        | 4                | 925.85                         |                            |
| 1        | 8                | 1844.66                        |                            |

#### Partial FC, sample_ratio=0.1

**batch_size=64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 245.12            |                 |
| 1        | 4                | 64                    | 945.44            |                 |
| 1        | 8                | 64                    | 1858.57           |                 |

**batch_size=max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bzs=115) | MXNet samples/s(max bzs=?) |
| -------- | ---------------- | ------------------------------ | -------------------------- |
| 1        | 1                | 248.01                         | 1.00                       |
| 1        | 4                | 973.63                         | 3.93                       |
| 1        | 8                | 1933.88                        | 7.8                        |



### Face Emore & Y1 & FP32 Thoughputs

#### Data Parallelism

**batch_size = 256**


| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 256                   | 1961.52           |                 |
| 1        | 4                | 256                   | 7354.49           |                 |
| 1        | 8                | 256                   | 14298.02          |                 |

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bzs=350) | MXNet samples/s(max bzs=352) |
| -------- | ---------------- | ------------------------------ | ---------------------------- |
| 1        | 1                | 1969.66                        |                              |
| 1        | 4                | 7511.53                        |                              |
| 1        | 8                | 14756.03                       |                              |

#### Model Parallelism

**batch_size = 256**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 256                   | 1963.62           | 984.2           |
| 1        | 4                | 256                   | 7264.54           | 984.88          |
| 1        | 8                | 256                   | 14049.75          | 1030.58         |

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bzs=400) | MXNet samples/s(max bzs=352) |
| -------- | ---------------- | ------------------------------ | ---------------------------- |
| 1        | 1                | 1969.65                        | 974.26                       |
| 1        | 4                | 7363.77                        | 1017.78                      |
| 1        | 8                | 14436.38                       | 1038.6                       |

### Max num_classses

| node_num | gpu_num_per_node | batch_size_per_device | FP16 | Model Parallel | Partial FC | OneFlow num_classes | MXNet   num_classes |
| -------- | ---------------- | --------------------- | ---- | -------------- | ---------- | ------------------- | ------------------- |
| 1        | 1                | 64                    | True | True           | True       | 2000000             | 2100000             |
| 1        | 8                | 64                    | True | True           | True       | 13500000            | 9900000             |



## Conclusion
 
The above series of tests show that:

1. With the increase of `batch_size_per_device`, the throughput of MXNet  hard to breakthrough even using Partial FC optimization while the throughput of OneFlow has always maintained a relatively stable linear growth

2. Under the same situation, OneFlow supports a larger scale of `batch_size` and `num_classes`. When using a batch size of 64 in one machine with 8 GPUs, optimization of FP16, model_parallel, and partial_fc.
The value of `num_classes` supported by OneFlow is 1.36 times of one supported by MXNet(13,500,000 vs 9,900,000).

For more data details, please check OneFlow and MXNet in DLPerf.