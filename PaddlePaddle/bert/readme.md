# 【DLPerf】PaddlePaddle BERT base测评

# Overview

本次复现采用了[PaddlePaddle官方仓库](https://github.com/PaddlePaddle/models/tree/release/1.8)中的[BERT](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT)，目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试仅覆盖 FP32 精度，后续将持续维护，增加混合精度训练，XLA 等多种方式的测评。



- [Overview](#overview)
- [Environment](#environment)
  * [系统](#--)
  * [框架](#--)
- [Quick Start](#quick-start)
  * [项目代码](#----)
  * [框架安装](#----)
  * [NCCL](#nccl)
  * [数据集](#---)
- [Training](#training)
  * [单机](#--)
  * [2机16卡](#2-16-)
  * [4机32卡](#4-32-)
- [Result](#result)
  * [吞吐率及加速比](#-------)
    + [计算规则](#----)
      - [1.测速脚本](#1----)
      - [2.均值速度和中值速度](#2---------)
      - [3.加速比以中值速度计算](#3----------)
  * [BERT-Base  batch size=32](#bert-base--batch-size-32)
    + [FP32 & Without XLA](#fp32---without-xla)
  * [BERT-Base  batch size=64](#bert-base--batch-size-64)
    + [FP32 & Without XLA](#fp32---without-xla-1)
  * [BERT-Base  batch size=96](#bert-base--batch-size-96)
    + [FP32 & Without XLA](#fp32---without-xla-2)
  * [完整日志](#----)



# Environment

## 系统

- 系统：Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- 显卡：Tesla V100-SXM2-16GB x 8
- 驱动：NVIDIA 440.33.01
- CUDA：10.2
- cuDNN：7.6.5
- NCCL：2.7.3

## 框架

- **paddle 1.8.3.post107**

# Quick Start

## 项目代码

- [PaddlePaddle官方仓库](https://github.com/PaddlePaddle/models/tree/release/1.8)
  - [BERT项目主页](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT)

下载官方源码：

```shell
git clone https://github.com/PaddlePaddle/models/tree/release/1.8
cd models/PaddleNLP/pretrain_language_models/BERT
```


将本页面scripts路径下的脚本：`make_pretrain_data.sh` 放入BERT/data路径下，其余脚本全部放入：BERT/路径下

## 框架安装

```shell
python3 -m pip install paddlepaddle-gpu==1.8.3.post107 -i https://mirror.baidu.com/pypi/simple
```

## NCCL

paddle的分布式训练底层依赖NCCL库，需要从[NVIDIA-NCCL官网下载](https://developer.nvidia.com/nccl/nccl-download)并安装和操作系统、CUDA版本适配的NCCL。
本次测试中安装2.7.3版本的NCCL：

```shell
sudo dpkg -i nccl-repo-ubuntu1604-2.7.3-ga-cuda10.2_1-1_amd64.deb
sudo apt update
sudo apt install libnccl2=2.7.3-1+cuda10.2 libnccl-dev=2.7.3-1+cuda10.2
```

## 数据集

本次BERT的预训练过程使用了paddle官方的示例数据集：[demo_wiki_train.gz](https://github.com/PaddlePaddle/models/blob/release/1.8/PaddleNLP/pretrain_language_models/BERT/data/train/demo_wiki_train.gz)，由于数据集规模较小，我们在此基础上制作了demo_wiki_train_50.gz用于预训练。数据集制作过程如下：

```shell
cd models/PaddleNLP/pretrain_language_models/BERT/data
bash make_pretrain_data.sh
```

脚本将复制demo_wiki_train的内容，构造出一个50倍数据规模的训练集demo_wiki_train_50.gz。

# Training

集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里设置batch size为32、64和96，分别在1机1卡～4机32卡的情况下进行了多组训练。

## 单机

`models/PaddleNLP/pretrain_language_models/BERT`目录下,执行脚本:

```shell
bash run_single_node.sh
```

对单机1卡、2卡、4卡、8卡分别做6组测试。单机多机脚本默认的batch size为32，可以通过参数指定，如指定batch size为64，`bash run_single_node.sh 64`，或96，`bash run_single_node.sh 96`。

## 2机16卡

2机、4机等多机情况下，需要在所有机器节点上相同路径准备同样的数据集、以完成分布式训练。


如2机：NODE1='10.11.0.2'     NODE2='10.11.0.3' 的训练，需在两台机器上分别准备好数据集后，NODE1节点`models/PaddleNLP/pretrain_language_models/BERT/`目录下,执行脚本:

```shell
bash run_two_node.sh
```

NODE2节点`models/PaddleNLP/pretrain_language_models/BERT/`目录下，修改run_two_node.sh脚本中的`CURRENT_NODE=$NODE2`，再执行`bash run_two_node.sh `，即可运行2机16卡的训练，同样默认测试6次。

## 4机32卡

流程同上，在4个机器节点上分别执行：

```shell
bash run_multi_node.sh
```

以运行4机32卡的训练，默认测试6组。

# Result

## 吞吐率及加速比

执行以下命令，即可计算各种测试配置下的吞吐率及加速比：

```shell
python extract_paddle_logs.py --log_dir=logs/paddle/bert/bz64 --batch_size_per_device=64
```

输出：

```shell
logs/paddle/bert/bz64/4n8g/bert_b64_fp32_4.log {4: 2743.19}
logs/paddle/bert/bz64/4n8g/bert_b64_fp32_1.log {4: 2743.19, 1: 2699.39}
logs/paddle/bert/bz64/4n8g/bert_b64_fp32_2.log {4: 2743.19, 1: 2699.39, 2: 2745.97}
logs/paddle/bert/bz64/4n8g/bert_b64_fp32_6.log {4: 2743.19, 1: 2699.39, 2: 2745.97, 6: 2687.66}
logs/paddle/bert/bz64/4n8g/bert_b64_fp32_3.log {4: 2743.19, 1: 2699.39, 2: 2745.97, 6: 2687.66, 3: 2730.36}
logs/paddle/bert/bz64/4n8g/bert_b64_fp32_5.log {4: 2743.19, 1: 2699.39, 2: 2745.97, 6: 2687.66, 3: 2730.36, 5: 2745.92}
logs/paddle/bert/bz64/1n8g/bert_b64_fp32_4.log {4: 780.47}
logs/paddle/bert/bz64/1n8g/bert_b64_fp32_1.log {4: 780.47, 1: 756.94}
logs/paddle/bert/bz64/1n8g/bert_b64_fp32_2.log {4: 780.47, 1: 756.94, 2: 765.51}
logs/paddle/bert/bz64/1n8g/bert_b64_fp32_6.log {4: 780.47, 1: 756.94, 2: 765.51, 6: 744.27}
logs/paddle/bert/bz64/1n8g/bert_b64_fp32_3.log {4: 780.47, 1: 756.94, 2: 765.51, 6: 744.27, 3: 769.89}
logs/paddle/bert/bz64/1n8g/bert_b64_fp32_5.log {4: 780.47, 1: 756.94, 2: 765.51, 6: 744.27, 3: 769.89, 5: 737.23}
logs/paddle/bert/bz64/1n4g/bert_b64_fp32_4.log {4: 436.65}
logs/paddle/bert/bz64/1n4g/bert_b64_fp32_1.log {4: 436.65, 1: 463.53}
logs/paddle/bert/bz64/1n4g/bert_b64_fp32_2.log {4: 436.65, 1: 463.53, 2: 462.61}
logs/paddle/bert/bz64/1n4g/bert_b64_fp32_6.log {4: 436.65, 1: 463.53, 2: 462.61, 6: 441.4}
logs/paddle/bert/bz64/1n4g/bert_b64_fp32_3.log {4: 436.65, 1: 463.53, 2: 462.61, 6: 441.4, 3: 424.21}
logs/paddle/bert/bz64/1n4g/bert_b64_fp32_5.log {4: 436.65, 1: 463.53, 2: 462.61, 6: 441.4, 3: 424.21, 5: 442.09}
logs/paddle/bert/bz64/1n1g/bert_b64_fp32_4.log {4: 137.2}
logs/paddle/bert/bz64/1n1g/bert_b64_fp32_1.log {4: 137.2, 1: 137.06}
logs/paddle/bert/bz64/1n1g/bert_b64_fp32_2.log {4: 137.2, 1: 137.06, 2: 137.18}
logs/paddle/bert/bz64/1n1g/bert_b64_fp32_6.log {4: 137.2, 1: 137.06, 2: 137.18, 6: 137.35}
logs/paddle/bert/bz64/1n1g/bert_b64_fp32_3.log {4: 137.2, 1: 137.06, 2: 137.18, 6: 137.35, 3: 137.39}
logs/paddle/bert/bz64/1n1g/bert_b64_fp32_5.log {4: 137.2, 1: 137.06, 2: 137.18, 6: 137.35, 3: 137.39, 5: 137.59}
logs/paddle/bert/bz64/1n2g/bert_b64_fp32_4.log {4: 251.44}
logs/paddle/bert/bz64/1n2g/bert_b64_fp32_1.log {4: 251.44, 1: 252.99}
logs/paddle/bert/bz64/1n2g/bert_b64_fp32_2.log {4: 251.44, 1: 252.99, 2: 254.32}
logs/paddle/bert/bz64/1n2g/bert_b64_fp32_6.log {4: 251.44, 1: 252.99, 2: 254.32, 6: 252.04}
logs/paddle/bert/bz64/1n2g/bert_b64_fp32_3.log {4: 251.44, 1: 252.99, 2: 254.32, 6: 252.04, 3: 252.72}
logs/paddle/bert/bz64/1n2g/bert_b64_fp32_5.log {4: 251.44, 1: 252.99, 2: 254.32, 6: 252.04, 3: 252.72, 5: 252.7}
logs/paddle/bert/bz64/2n8g/bert_b64_fp32_4.log {4: 1418.26}
logs/paddle/bert/bz64/2n8g/bert_b64_fp32_1.log {4: 1418.26, 1: 1441.44}
logs/paddle/bert/bz64/2n8g/bert_b64_fp32_2.log {4: 1418.26, 1: 1441.44, 2: 1431.65}
logs/paddle/bert/bz64/2n8g/bert_b64_fp32_6.log {4: 1418.26, 1: 1441.44, 2: 1431.65, 6: 1389.89}
logs/paddle/bert/bz64/2n8g/bert_b64_fp32_3.log {4: 1418.26, 1: 1441.44, 2: 1431.65, 6: 1389.89, 3: 1447.72}
logs/paddle/bert/bz64/2n8g/bert_b64_fp32_5.log {4: 1418.26, 1: 1441.44, 2: 1431.65, 6: 1389.89, 3: 1447.72, 5: 1421.38}
{'bert': {'1n1g': {'average_speed': 137.29,
                   'batch_size_per_device': 64,
                   'median_speed': 137.27,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 252.7,
                   'batch_size_per_device': 64,
                   'median_speed': 252.71,
                   'speedup': 1.84},
          '1n4g': {'average_speed': 445.08,
                   'batch_size_per_device': 64,
                   'median_speed': 441.74,
                   'speedup': 3.22},
          '1n8g': {'average_speed': 759.05,
                   'batch_size_per_device': 64,
                   'median_speed': 761.22,
                   'speedup': 5.55},
          '2n8g': {'average_speed': 1425.06,
                   'batch_size_per_device': 64,
                   'median_speed': 1426.52,
                   'speedup': 10.39},
          '4n8g': {'average_speed': 2725.42,
                   'batch_size_per_device': 64,
                   'median_speed': 2736.78,
                   'speedup': 19.94}}}
Saving result to ./result/bz64_result.json
```

### 计算规则

#### 1.测速脚本

- extract_paddle_logs.py

extract_paddle_logs.py根据官方在log中打印的速度，在120个iter中，排除前20iter，取后100个iter的速度做平均；

#### 2.均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行6次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度。

#### 3.加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5

## BERT-Base  batch size=32

### FP32 & Without XLA

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 132.64    | 1.00    |
| 1        | 2       | 228.12    | 1.72    |
| 1        | 4       | 406.02    | 3.06    |
| 1        | 8       | 615.12    | 4.64    |
| 2        | 16      | 1116.02   | 8.41    |
| 4        | 32      | 2073.6    | 15.63   |

## BERT-Base  batch size=64

### FP32 & Without XLA

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 137.27    | 1.00    |
| 1        | 2       | 252.71    | 1.84    |
| 1        | 4       | 441.74    | 3.22    |
| 1        | 8       | 761.22    | 5.55    |
| 2        | 16      | 1426.52   | 10.39   |
| 4        | 32      | 2736.78   | 19.94   |

## BERT-Base  batch size=96

### FP32 & Without XLA

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 136.97    | 1.00    |
| 1        | 2       | 258.73    | 1.89    |
| 1        | 4       | 490.38    | 3.58    |
| 1        | 8       | 868.6     | 6.34    |
| 2        | 16      | 1631.36   | 11.91   |
| 4        | 32      | 3167.68   | 23.13   |

## 完整日志

[bert.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/PaddlePaddle/bert.zip)
