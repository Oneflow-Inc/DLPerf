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



## 快速开始 Quick Start


### 1. 前期准备


- #### 数据集

OFRecord 数据集是 **OFRecord 文件的集合** 。将多个 OFRecord 文件，按照 OneFlow 约定的文件名格式，存放在同一个目录中，就得到了 OFRecord 数据集。

默认情况下，OFRecord 数据集目录中的文件，统一以 `part-xxx` 的方式命名，其中的 "xxx" 是从 0 开始的文件编号，有补齐和不补齐两种选择。

以下是没有采用补齐的命名风格示例：

```
mnist_kaggle/train/
├── part-0
├── part-1
├── part-10
├── part-11
├── part-12
├── part-13
├── part-14
├── part-15
├── part-2
├── part-3
├── part-4
├── part-5
├── part-6
├── part-7
├── part-8
└── part-9
```



以下是有补齐的命名风格：

```
mnist_kaggle/train/
├── part-00000
├── part-00001
├── part-00002
├── part-00003
├── part-00004
├── part-00005
├── part-00006
├── part-00007
├── part-00008
├── part-00009
├── part-00010
├── part-00011
├── part-00012
├── part-00013
├── part-00014
├── part-00015
```

OneFlow 采用此约定，与 Spark 默认存储的文件名一致，方便使用 Spark 制作与转化 OFRecord 数据。

实际上，文件名前缀（`part-`）、文件名编号是否补齐、按多少位补齐，均可以自行指定，只需要在加载数据集时，保持相关参数一致即可。

OneFlow 提供了加载 OFRecord 数据集的接口，使得我们只要指定数据集目录的路径，就可以享受 OneFlow 框架所带来的多线程、数据流水线等优势。

准备 Face Emore 和 Glint360k 的 OFReocord 数据集，可以选择根据 [加载与准备 OFRecord 数据集](https://docs.oneflow.org/extended_topics/how_to_make_ofdataset.html)文档中 Python 脚本生成所有数据的完整 OFRecord + Spark Shuffle + Spark Partition 的方式，也可以选择只使用 Python 脚本生成多块 OFRecord 的方式，用以进行 InsightFace 的测试。

1. Python 脚本直接生成

运行 tools/dataset_convert/mx_recordio_2_ofrecord_shuffled_npart.py

```
python tools/dataset_convert/mx_recordio_2_ofrecord_shuffled_npart.py  --data_dir /data/face_test/dataset/faces_emore  --output_filepath ofrecord/train --num_part 16
```

屏幕打印

```
Converting images: 5790000 of 5822653
Converting images: 5800000 of 5822653
Converting images: 5810000 of 5822653
Converting images: 5820000 of 5822653
```

完成后，即可直接生成对应 `num_part` 数量的 OFRecord。以生成 16 个 part 为例：

```
$ tree ofrecord/test/
ofrecord/test/
|-- _SUCCESS
|-- part-00000
|-- part-00001
|-- part-00002
|-- part-00003
|-- part-00004
|-- part-00005
|-- part-00006
|-- part-00007
|-- part-00008
|-- part-00009
|-- part-00010
|-- part-00011
|-- part-00012
|-- part-00013
|-- part-00014
`-- part-00015

0 directories, 17 files
```



2. Python 脚本 + Spark Shuffle + Spark Partition

运行 [tools/dataset_convert/mx_recordio_2_ofrecord.py](https://github.com/Oneflow-Inc/oneflow_face/tree/b7207f29f4e9254b1d9ea748a628d223b9aafd1a) 生成所有数据的完整 OFRecord（`part-0`）

```
python tools/dataset_convert/mx_recordio_2_ofrecord.py --data_dir /data/face_test/dataset/faces_emore  --output_filepath ofrecord/train 
```

安装部署 Spark 环境后，输入 Spark 命令

```scala
//Spark 启动命令：
./Spark-2.4.3-bin-hadoop2.7/bin/Spark-shell --jars ~/Spark-oneflow-connector-assembly-0.1.0.jar --driver-memory=64G --conf Spark.local.dir=/tmp/
// shuffle 并分成 96 个 part 的执行命令：
import org.oneflow.Spark.functions._
Spark.read.chunk("data_path").shuffle().repartition(96).write.chunk("new_data_path")
sc.formatFilenameAsOneflowStyle("new_data_path")
```

- #### 网络对齐

OneFlow 的实现与 MXNet 进行了严格对齐，主要包括：

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

将本仓库 DLPerf/OneFlow/Recognition/InsightFace/scripts 路径源码放至 oneflow_face 路径下，使用 /scripts 路径下的 insightface_train.py 替换 oneflow_face 路径下的文件

```
mv scripts/insightface_train.py oneflow_face/insightface_train.py
```
以测试 FP32 数据并行 r100，Face emore 数据集搭配 arcface loss，batch_size_per_device=64, 单机 8 卡的真实数据 150 batch 为例，运行
```
bash run_single_node.sh r100 emore arcface 64 batch 150 fp32 False False
```
换成相同配置模型并行， 输入
```
bash run_single_node.sh r100 emore arcface 64 batch 150 fp32 True False
```
换成 Partial FC 优化，输入
```
bash run_single_node.sh r100 emore arcface 64 batch 150 fp32 True True
```
如果希望测试 Partial FC 优化 r100 Glint360k 数据集搭配 cosface loss，其他配置不变，则输入
```
bash run_single_node.sh r100_glint360k glint360k cosface 64 batch 150 fp32 True True
```

即可针对不同网络和配置，对单机单卡、2 卡、4 卡、8 卡等情况进行测试，并将 log 信息保存在当前目录的对应测评配置路径中。

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
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 64                    | 245.0              | 1.00    |
| 1        | 4                | 64                    | 923.23             | 3.77    |
| 1        | 8                | 64                    | 1836.8             | 7.5     |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 96                    | 252.76             | 1.00    |
| 1        | 4                | 96                    | 969.27             | 3.83    |
| 1        | 8                | 96                    | 1925.6             | 7.62    |

#### Model Parallelism

**batch_size = 64**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 64                    | 245.29             | 1.00    |
| 1        | 4                | 64                    | 938.83             | 3.83    |
| 1        | 8                | 64                    | 1854.15            | 7.55    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 115                   | 245.92             | 1.00    |
| 1        | 4                | 115                   | 968.72             | 3.94    |
| 1        | 8                | 115                   | 1925.59            | 7.83    |

#### Partial FC, sample_ratio=0.1

**batch_size=64**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 64                    | 247.97             | 1.00    |
| 1        | 4                | 64                    | 946.54             | 3.82    |
| 1        | 8                | 64                    | 1864.31            | 7.48    |

**batch_size=max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 120                   | 256.61             | 1.00    |
| 1        | 4                | 120                   | 990.82             | 3.86    |
| 1        | 8                | 120                   | 1962.76            | 7.65    |



### Glint360k & R100 & FP32 

#### Data Parallelism

**batch_size = 64**


| node_num | gpu_num_per_node | batch_size_per_device | samples/s| speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 64                    | 230.22             | 1.00    |
| 1        | 4                | 64                    | 847.71             | 3.68    |
| 1        | 8                | 64                    | 1688.62            | 7.33    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s| speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 85                    | 229.94             | 1.00    |
| 1        | 4                | 85                    | 856.61             | 3.73    |
| 1        | 8                | 85                    | 1707.03            | 7.42    |

#### Model Parallelism

**batch_size = 64**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 64                    | 230.33             | 1.00    |
| 1        | 4                | 64                    | 912.24             | 3.96    |
| 1        | 8                | 64                    | 1808.27            | 7.85    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 100                   | 231.86             | 1.00    |
| 1        | 4                | 100                   | 925.85             | 3.99    |
| 1        | 8                | 100                   | 1844.66            | 7.96    |

#### Partial FC, sample_ratio=0.1

**batch_size=64**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 64                    | 245.12             | 1.00    |
| 1        | 4                | 64                    | 945.44             | 3.86    |
| 1        | 8                | 64                    | 1858.57            | 7.58    |

**batch_size=max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 115                   | 248.01             | 1.00    |
| 1        | 4                | 115                   | 973.63             | 3.93    |
| 1        | 8                | 115                   | 1933.88            | 7.8     |



### Face Emore & Y1 & FP32

#### Data Parallelism

**batch_size = 256**


| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 256                   | 1961.52            | 1.00    |
| 1        | 4                | 256                   | 7354.49            | 3.75    |
| 1        | 8                | 256                   | 14298.02           | 7.29    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 350                   | 1969.66            | 1.00    |
| 1        | 4                | 350                   | 7511.53            | 3.81    |
| 1        | 8                | 350                   | 14756.03           | 7.49    |

#### Model Parallelism

**batch_size = 256**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 256                   | 1963.62            | 1.00    |
| 1        | 4                | 256                   | 7264.54            | 3.7     |
| 1        | 8                | 256                   | 14049.75           | 7.16    |

**batch_size = max**

| node_num | gpu_num_per_node | batch_size_per_device | samples/s | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 400                   | 1969.65            | 1.00    |
| 1        | 4                | 400                   | 7363.77            | 3.74    |
| 1        | 8                | 400                   | 14436.38           | 7.33    |

### Max num_classes

| node_num | gpu_num_per_node | batch_size_per_device | fp16 | Model Parallel | Partial FC | num_classes |
| -------- | ---------------- | --------------------- | ---- | -------------- | ---------- | ----------- |
| 1        | 1                | 64                    | True | True           | True       | 2000000     |
| 1        | 8                | 64                    | True | True           | True       | 13500000    |

目前 InsightFace 的相关代码及结果已经 PR 至 [insightface](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487)/[recognition](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487/recognition)/[**oneflow_face**](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487/recognition/oneflow_face)

[deepinsight](https://github.com/deepinsight)/[insightface](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487) 官方测评结果详见 [Benchmark](https://github.com/deepinsight/insightface/tree/79aacd2bb3323fa50a125b828bb1656166604487/recognition/partial_fc#benchmark)。

详细 Log 信息可下载：[insightface_fp32_logs.tar](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/InsightFace/insightface_fp32_logs.tar)。
