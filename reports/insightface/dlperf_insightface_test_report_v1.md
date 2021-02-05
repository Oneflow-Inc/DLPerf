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
      - [Partial FC, sample_ratio = 0.1](#partial-fc-sample_ratio--01)
    - [Glint360k & R100 & FP32 Thoughputs](#glint360k--r100--fp32-thoughputs)
      - [Data Parallelism](#data-parallelism-1)
      - [Model Parallelism](#model-parallelism-1)
      - [Partial FC, sample_ratio = 0.1](#partial-fc-sample_ratio--01-1)
    - [Face Emore & Y1 & FP32 Thoughputs](#face-emore--y1--fp32-thoughputs)
      - [Data Parallelism](#data-parallelism-2)
      - [Model Parallelism](#model-parallelism-2)
    - [Max num_classes](#max-num_classes)
  - [Conclusion](#conclusion)
## Data

Reproduction procedures, introductions, logs, data, and English reports could be fetched in DLPerf repository: https://github.com/Oneflow-Inc/DLPerf


## Frameworks & MOdels

| Framework                                                          | Version              | Source                                                                                                            |
| ------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.3.4) | 0.3.4             | [oneflow_face]()                                                                                                    |
| [deepinsight](https://github.com/deepinsight)                 | 2021-01-20 update | [deepinsight/insightface](https://github.com/deepinsight/insightface/tree/a9beb60971fb8115698859c35fdca721d6f75f5d) |

## Configration

### 1. Network Alignment 

rigorous alignment has been completed between OneFlow and MxNet, including:


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

In this report, batch size means the number of samples on each device(GPU), bsz (batch size per GPU) in short. In the tests, it will give the static value or maximum of batch size with different numbers of GPU tests in different frameworks.

### 3. Num Classes 

In this report, num classes mean the number of face categories. In the tests, it will give the static value or maximum of num classes with different numbers of GPU tests in different frameworks.

## Results

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
| 1        | 1                | 252.76                          | 288.0                       |
| 1        | 4                | 969.27                          | 733.1                       |
| 1        | 8                | 1925.6                          | 749.42                      |

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
| 1        | 1                | 245.92                         | 242.2                         |
| 1        | 4                | 968.72                         | 724.26                        |
| 1        | 8                | 1925.59                        | 821.06                        |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/emore_r100_fp32_bmax_mp_en.png)

#### Partial FC, sample_ratio = 0.1

**batch_size=64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 247.97            | 223.11          |
| 1        | 4                | 64                    | 946.54            | 799.19          |
| 1        | 8                | 64                    | 1864.31           | 1586.09         |

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
| 1        | 1                | 64                    | 230.22            | OOM             |
| 1        | 4                | 64                    | 847.71            | OOM             |
| 1        | 8                | 64                    | 1688.62           | OOM             |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/glint360k_r100_fp32_b64_dp_en.png)

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bsz=85) | MXNet samples/s(max bsz=?) |
| -------- | ---------------- | ----------------------------- | -------------------------- |
| 1        | 1                | 229.94                        | OOM                        |
| 1        | 4                | 856.61                        | OOM                        |
| 1        | 8                | 1707.03                       | OOM                        |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/glint360k_r100_fp32_bmax_dp_en.png)

#### Model Parallelism

**batch_size = 64**

| node_num | gpu_num_per_node | batch_size_per_device | OneFlow samples/s | MXNet samples/s |
| -------- | ---------------- | --------------------- | ----------------- | --------------- |
| 1        | 1                | 64                    | 230.33            | OOM             |
| 1        | 4                | 64                    | 912.24            | OOM             |
| 1        | 8                | 64                    | 1808.27           | OOM             |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/glint360k_r100_fp32_b64_mp_en.png)

**batch_size = max**

| node_num | gpu_num_per_node | OneFlow samples/s(max bsz=100) | MXNet samples/s(max bsz=?) |
| -------- | ---------------- | ------------------------------ | -------------------------- |
| 1        | 1                | 231.86                         | OOM                        |
| 1        | 4                | 925.85                         | OOM                        |
| 1        | 8                | 1844.66                        | OOM                        |

![ ](https://github.com/Oneflow-Inc/DLPerf/blob/dev_sx_insightface/reports/imgs/glint360k_r100_fp32_bmax_mp_en.png)

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

### Max num_classes

| node_num | gpu_num_per_node | batch_size_per_device | FP16 | Model Parallel | Partial FC | OneFlow num_classes | MXNet   num_classes |
| -------- | ---------------- | --------------------- | ---- | -------------- | ---------- | ------------------- | ------------------- |
| 1        | 1                | 64                    | True | True           | True       | 2000000             | 1800000             |
| 1        | 8                | 64                    | True | True           | True       | 13500000            | 12000000            |



## Conclusion
 
The above series of tests show that:

1. When dealing with the very large datasets, such as Glint360k, MXNet will meet OOM(out of memory), and fail to run with data parallelism or model parallelism, unless it uses Partial FC.
2. With the increase of `batch_size_per_device`, the throughput of MXNet hard to breakthrough even using Partial FC optimization while the throughput of OneFlow has always maintained a relatively stable linear growth.

3. Under the same situation, OneFlow supports a larger scale of `batch_size` and `num_classes`. When using a batch size of 64 in one machine with 8 GPUs, optimization of FP16, model_parallel, and partial_fc.
The value of `num_classes` supported by OneFlow is 1.36 times of one supported by MXNet(13,500,000 vs 9,900,000).

For more data details, please check OneFlow and MXNet in DLPerf.
