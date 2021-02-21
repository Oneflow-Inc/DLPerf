# OneFlow InsightFace 测评


## 概述 Overview


本测试基于 OneFlow [oneflow_face](https://github.com/Oneflow-Inc/oneflow_face/tree/master) 提供与 [deepinsight](https://github.com/deepinsight)/**[insightface](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487)** 仓库中 MXNet 等价实现的 InsightFace 网络，进行单机单卡、单机多卡的速度评测，评判框架在分布式训练情况下的横向拓展能力。


目前，该测试覆盖 FP32 精度，后续将持续维护，增加使用其他优化方式的测评。


## 内容目录 Table Of Content



- [OneFlow InsightFace 测评](#oneflow-insightface-测评)
	- [概述 Overview](#概述-overview)
	- [内容目录 Table Of Content](#内容目录-table-of-content)
	- [环境 Environment](#环境-environment)
		- [系统](#系统)
			- [Feature support matrix](#feature-support-matrix)
		- [模型配置](#模型配置)
	- [快速开始 Quick Start](#快速开始-quick-start)
		- [1. 前期准备](#1-前期准备)
		- [2. 运行测试](#2-运行测试)
		- [3. 数据处理](#3-数据处理)
	- [性能结果 Performance](#性能结果-performance)
		- [Face Emore & R100 & FP32](#face-emore--r100--fp32)
			- [Data Parallelism](#data-parallelism)
			- [Model Parallelism](#model-parallelism)
			- [Partial FC, sample_ratio=0.1](#partial-fc-sample_ratio01)
		- [Glint360k & R100 & FP32](#glint360k--r100--fp32)
			- [Data Parallelism](#data-parallelism-1)
			- [Model Parallelism](#model-parallelism-1)
			- [Partial FC, sample_ratio=0.1](#partial-fc-sample_ratio01-1)
		- [Face Emore & Y1 & FP32](#face-emore--y1--fp32)
			- [Data Parallelism](#data-parallelism-2)
			- [Model Parallelism](#model-parallelism-2)
		- [Max num_classes](#max-num_classes)


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


### 模型配置

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


每个节点有 8 张 V100 显卡， 每张显卡显存 16 G。目前的单机测试仅使用其中一个节点。


- #### 单机测试


在节点 1 的容器内下载 oneflow_face 源码和本仓库源码：


````
git clone https://github.com/Oneflow-Inc/oneflow_face.git
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 DLPerf/OneFlow/Recognition/InsightFace/scripts 路径源码放至 oneflow_face 路径下，使用 scripts 路径下的 insightface_train.py 替换 oneflow_face 路径下的文件
```
mv scripts/insightface_train.py oneflow_face/insightface_train.py
```
同时根据测试需求打开 `run_single_node.sh` 文件，修改脚本参数。以测试 FP32 数据并行的 r100，Face emore 数据集搭配 arcface loss，batch_size_per_device=64, 单机 8 卡的真实数据 150 batch 为例，确认参数

```
# run_single_node.sh 
workspace=${1:-"/path/to/workspace/"} 
network=${2:-"r100"}	# optional: r100_glint360k, y1
dataset=${3:-"emore"}	# optional: glint360k_8GPU
loss=${4:-"arcface"}	# optional: softmax, cosface, combined
bz_per_device=${5:-64}	
train_unit=${6:-"batch"}	# optional: epoch
iter_num=${7:-150}	
precision=${8:-fp32}	# optional: fp16
model_parallel=${9:-False}	# optional: True
partila_fc=${10:-False}		# optional: True
sample_ratio=${11:-0.1}
num_classes=${12:-85744}	# default num classes in the face emore dataset
use_synthetic_data=${13:-False}		# optional: True
```

保存配置，直接运行
```
bash run_single_node.sh 
```

即可针对当前网络和配置进行测试。

修改配置之后，即可对不同网络和配置，单机单卡、2 卡、4 卡、8 卡等情况进行测试，并将 log 信息保存在当前目录的对应测评配置路径中。

- #### 寻找最大 `num_classes`

寻找 OneFlow InsightFace 可支持的最大 `num_classes`，测试时需要使用合成数据 `use_synthetic_data=True`，并修改 train_emore.sh 的 `num_classes`。

以使用 ResNet100 和 Face Emore 作为 Backbone 和数据集，测试合成数据单机 8 卡，`batch_size_per_device=64`，`model_parallel=True`，`partial_fc=True`， `num_classes=1500000` 训练 150 个 batch 为例，运行
```
bash train_emore.sh ${workspace_path} r100 emore arcface 1 64 batch 150 8 fp16 1 1 1 150000 True
```
若可以正常运行打印 log，即说明可支持当前 `num_classes` 设置，可尝试更大的 `num_classes`。

### 3. 数据处理


测试进行了多组训练（本测试中取 5 次），每次训练过程取第 1 个 epoch 的前 150 iter，计算训练速度时取后 100 iter 的数据，以降低抖动。最后将 5 次训练的结果取中位数得到最终速度，并最终以此数据计算加速比。


运行 DLPerf/OneFlow/Recognition/InsightFace/extract_pytorch_logs_time.py，即可得到针对不同配置测试结果 log 数据处理的结果： 


```
python extract_oneflow_logs_time.py -ld r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/ 
```

结果打印如下


```
$ python extract_oneflow_logs_time.py -ld r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n1g/r100_b115_fp32_1.log {1: 247.0}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n1g/r100_b115_fp32_5.log {1: 247.0, 5: 246.1}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n1g/r100_b115_fp32_3.log {1: 247.0, 5: 246.1, 3: 245.67}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n1g/r100_b115_fp32_2.log {1: 247.0, 5: 246.1, 3: 245.67, 2: 245.29}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n1g/r100_b115_fp32_4.log {1: 247.0, 5: 246.1, 3: 245.67, 2: 245.29, 4: 245.92}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n4g/r100_b115_fp32_1.log {1: 962.97}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n4g/r100_b115_fp32_5.log {1: 962.97, 5: 970.73}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n4g/r100_b115_fp32_3.log {1: 962.97, 5: 970.73, 3: 968.72}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n4g/r100_b115_fp32_2.log {1: 962.97, 5: 970.73, 3: 968.72, 2: 967.52}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n4g/r100_b115_fp32_4.log {1: 962.97, 5: 970.73, 3: 968.72, 2: 967.52, 4: 969.25}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n8g/r100_b115_fp32_1.log {1: 1927.4}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n8g/r100_b115_fp32_5.log {1: 1927.4, 5: 1924.83}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n8g/r100_b115_fp32_3.log {1: 1927.4, 5: 1924.83, 3: 1927.21}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n8g/r100_b115_fp32_2.log {1: 1927.4, 5: 1924.83, 3: 1927.21, 2: 1925.59}
r100_fp32_b115_oneflow_model_parallel_True_partial_fc_False_0.1/r100/1n8g/r100_b115_fp32_4.log {1: 1927.4, 5: 1924.83, 3: 1927.21, 2: 1925.59, 4: 1913.58}
{'r100': {'1n1g': {'average_speed': 246.0,
                   'batch_size_per_device': 115,
                   'median_speed': 245.92,
                   'speedup': 1.0},
          '1n4g': {'average_speed': 967.84,
                   'batch_size_per_device': 115,
                   'median_speed': 968.72,
                   'speedup': 3.94},
          '1n8g': {'average_speed': 1923.72,
                   'batch_size_per_device': 115,
                   'median_speed': 1925.59,
                   'speedup': 7.83}}}
Saving result to ./result/_result.json
```



## 性能结果 Performance


该小节提供针对 OneFlow 框架的 InsightFace 模型单机测试的性能结果和完整 log 日志。




### Face Emore & R100 & FP32 

#### Data Parallelism

**batch_size = 64**


| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 64                    | 245.0     | 1.00    |
| 1        | 4                | 64                    | 923.23    | 3.77    |
| 1        | 8                | 64                    | 1836.8    | 7.5     |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 96                    | 250.71    | 1.00    |
| 1        | 4                | 96                    | 972.8     | 3.88    |
| 1        | 8                | 96                    | 1931.76   | 7.71    |

#### Model Parallelism

**batch_size = 64**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 64                    | 245.29    | 1.00    |
| 1        | 4                | 64                    | 938.83    | 3.83    |
| 1        | 8                | 64                    | 1854.15   | 7.55    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 115                   | 246.55    | 1.00    |
| 1        | 4                | 115                   | 970.1     | 3.93    |
| 1        | 8                | 115                   | 1921.87   | 7.8     |

#### Partial FC, sample_ratio=0.1

**batch_size=64**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 64                    | 246.45    | 1.00    |
| 1        | 4                | 64                    | 948.96     | 3.85    |
| 1        | 8                | 64                    | 1872.81   | 7.6     |
| 2        | 8                | 64                    | 3540.09   | 14.36     |
| 4        | 8                | 64                   | 6931.6   | 28.13    |

**batch_size=max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 120                   | 256.61    | 1.00    |
| 1        | 4                | 120                   | 990.82    | 3.86    |
| 1        | 8                | 120                   | 1962.76   | 7.65    |
| 2        | 8                | 120                   | 3856.52   | 15.03    |
| 4        | 8                | 120                   | 7564.74   | 29.48    |

### Glint360k & R100 & FP32 

#### Data Parallelism

**batch_size = 64**


| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 64                    | 230.22    | 1.00    |
| 1        | 4                | 64                    | 847.71    | 3.68    |
| 1        | 8                | 64                    | 1688.62   | 7.33    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 85                    | 229.94    | 1.00    |
| 1        | 4                | 85                    | 856.61    | 3.73    |
| 1        | 8                | 85                    | 1707.03   | 7.42    |

#### Model Parallelism

**batch_size = 64**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 64                    | 230.33    | 1.00    |
| 1        | 4                | 64                    | 912.24    | 3.96    |
| 1        | 8                | 64                    | 1808.27   | 7.85    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 100                   | 231.86    | 1.00    |
| 1        | 4                | 100                   | 925.85    | 3.99    |
| 1        | 8                | 100                   | 1844.66   | 7.96    |

#### Partial FC, sample_ratio=0.1

**batch_size=64**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 64                    | 245.12    | 1.00    |
| 1        | 4                | 64                    | 945.44    | 3.86    |
| 1        | 8                | 64                    | 1858.57   | 7.58    |

**batch_size=max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 115                   | 248.01    | 1.00    |
| 1        | 4                | 115                   | 973.63    | 3.93    |
| 1        | 8                | 115                   | 1933.88   | 7.79    |



### Face Emore & Y1 & FP32

#### Data Parallelism

**batch_size = 256**


| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 256                   | 1961.52   | 1.00    |
| 1        | 4                | 256                   | 7354.49   | 3.75    |
| 1        | 8                | 256                   | 14298.02  | 7.29    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 350                   | 1969.66   | 1.00    |
| 1        | 4                | 350                   | 7511.53   | 3.81    |
| 1        | 8                | 350                   | 14756.03  | 7.49    |

#### Model Parallelism

**batch_size = 256**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 256                   | 1963.62   | 1.00    |
| 1        | 4                | 256                   | 7264.54   | 3.7     |
| 1        | 8                | 256                   | 14049.75  | 7.16    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | --------- | ------- |
| 1        | 1                | 400                   | 1969.65   | 1.00    |
| 1        | 4                | 400                   | 7363.77   | 3.74    |
| 1        | 8                | 400                   | 14436.38  | 7.33    |

### Max num_classes

| node_num | gpu_num_per_node | batch_size_per_device | fp16 | Model Parallel | Partial FC | num_classes |
| -------- | ---------------- | --------------------- | ---- | -------------- | ---------- | ----------- |
| 1        | 1                | 64                    | True | True           | True       | 2000000     |
| 1        | 8                | 64                    | True | True           | True       | 13500000    |

目前 InsightFace 的相关代码及结果已经 PR 至 [insightface](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487)/[recognition](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487/recognition)/[**oneflow_face**](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487/recognition/oneflow_face)

[deepinsight](https://github.com/deepinsight)/[insightface](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487) 官方测评结果详见 [Benchmark](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487/recognition/partial_fc#benchmark)。

详细 Log 信息可下载：[insightface_fp32_logs.tar](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/InsightFace/insightface_fp32_logs.tar)。
