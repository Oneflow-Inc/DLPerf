# Overview

本次复现采用了[PaddlePaddle-PLSC官方仓库](https://github.com/PaddlePaddle/PLSC/tree/9bba9c90f542a5e1d8e6d461fcd1f6af40da0918)中的paddle版的arcface人脸分类模型，目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试覆盖了FP32精度，后续将持续维护，增加更多方式的测评。



# Environment

## 系统

- 系统：Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- 显卡：Tesla V100-SXM2-16GB x 8
- 驱动：NVIDIA 440.33.01
- CUDA：10.2
- cuDNN：7.6.5
- NCCL：2.7.3

## 框架

- **paddlepaddle-gpu==1.8.5.post107**

## Feature support matrix

| Feature                       | Paddle |
| ----------------------------- | ------ |
| Multi-node,multi-gpu training | Yes    |
| NVIDIA NCCL                   | Yes    |

# Quick Start

## 项目代码

- [PaddlePaddle-PLSC官方仓库](https://github.com/PaddlePaddle/PLSC/tree/9bba9c90f542a5e1d8e6d461fcd1f6af40da0918)

下载官方源码：

```shell
git clone https://github.com/PaddlePaddle/PLSC.git
cd PLSC
git checkout 9bba9c90f542a5e1d8e6d461fcd1f6af40da0918
```

将本页面中scripts文件夹中的脚本和代码全部放入：`PLSC/`路径下。

修改PLSC/plsc/entry.py，在[Line:984](https://github.com/PaddlePaddle/PLSC/blob/9bba9c90f542a5e1d8e6d461fcd1f6af40da0918/plsc/entry.py#L984)下加入代码：

```python
local_train_info = [[], [], [], []]
    if batch_id==150:
        exit()
```

以使测试进行150 iter 后自动退出。



## 依赖安装


### 环境

1.本测试使用 conda 环境， 可以在使用如下命令创建plsc环境


```
conda env create -f environment.yaml
```

### NCCL

paddle的分布式训练底层依赖NCCL库，需要从[NVIDIA-NCCL官网下载](https://developer.nvidia.com/nccl/nccl-download)并安装和操作系统、CUDA版本适配的NCCL。本次测试中安装2.7.3版本的NCCL：

```shell
sudo dpkg -i nccl-repo-ubuntu1604-2.7.3-ga-cuda10.2_1-1_amd64.deb
sudo apt update
sudo apt install libnccl2=2.7.3-1+cuda10.2 libnccl-dev=2.7.3-1+cuda10.2
```

## 数据集

本次训练使用虚拟合成数据，无需准备数据集，如需真实训练数据集准备过程可参考[官方README](https://github.com/PaddlePaddle/PLSC/blob/master/docs/source/md/quick_start.md#%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)


# Training

集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里默认使用resnet50的backbone，batch size设为128，分别在1机1卡～4机32卡的情况下进行了多组训练。

## 单机

`PLSC/`目录下,执行脚本：

```shell
bash run_single_node.sh
```

对单机1卡、4卡、8卡分别做5组测试，默认测试fp32精度，batch_size=128。

## 2机16卡

2机、4机等多机情况下，需要在所有机器节点上相同路径准备同样的代码和脚本以完成分布式训练。

如2机：NODE1='10.11.0.2'     NODE2='10.11.0.3' 的训练，需在NODE1节点`PLSC/`目录下执行脚本:

```shell
bash run_two_node.sh r50 128 fp32 10.11.0.2 5
```

NODE2节点`PLSC/`目录下，执行：

```shell
bash run_two_node.sh r50 128 fp32 10.11.0.3 5
```

## 4机32卡

流程同上，在4个机器节点上分别执行：

```shell
bash run_two_node.sh r50 128 fp32 $NODE 5
```

# Result

## 吞吐率及加速比

执行以下命令，即可计算各种测试配置下的吞吐率及加速比：

```shell
python extract_paddle_logs.py  --log_dir=logs/paddle-plsc/arcface/bz128 --batch_size_per_device=128
```

输出：

```shell
logs/paddle-plsc/arcface/bz128/4n8g/r50_b128_fp32_1.log {1: 11165.17}
logs/paddle-plsc/arcface/bz128/4n8g/r50_b128_fp32_4.log {1: 11165.17, 4: 11077.58}
logs/paddle-plsc/arcface/bz128/4n8g/r50_b128_fp32_2.log {1: 11165.17, 4: 11077.58, 2: 11109.91}
logs/paddle-plsc/arcface/bz128/4n8g/r50_b128_fp32_3.log {1: 11165.17, 4: 11077.58, 2: 11109.91, 3: 11084.53}
logs/paddle-plsc/arcface/bz128/4n8g/r50_b128_fp32_5.log {1: 11165.17, 4: 11077.58, 2: 11109.91, 3: 11084.53, 5: 10997.67}
logs/paddle-plsc/arcface/bz128/1n8g/r50_b128_fp32_1.log {1: 2536.04}
logs/paddle-plsc/arcface/bz128/1n8g/r50_b128_fp32_4.log {1: 2536.04, 4: 2554.13}
logs/paddle-plsc/arcface/bz128/1n8g/r50_b128_fp32_2.log {1: 2536.04, 4: 2554.13, 2: 2545.3}
logs/paddle-plsc/arcface/bz128/1n8g/r50_b128_fp32_3.log {1: 2536.04, 4: 2554.13, 2: 2545.3, 3: 2563.28}
logs/paddle-plsc/arcface/bz128/1n8g/r50_b128_fp32_5.log {1: 2536.04, 4: 2554.13, 2: 2545.3, 3: 2563.28, 5: 2542.14}
logs/paddle-plsc/arcface/bz128/1n4g/r50_b128_fp32_1.log {1: 1555.18}
logs/paddle-plsc/arcface/bz128/1n4g/r50_b128_fp32_4.log {1: 1555.18, 4: 1539.66}
logs/paddle-plsc/arcface/bz128/1n4g/r50_b128_fp32_2.log {1: 1555.18, 4: 1539.66, 2: 1534.62}
logs/paddle-plsc/arcface/bz128/1n4g/r50_b128_fp32_3.log {1: 1555.18, 4: 1539.66, 2: 1534.62, 3: 1540.66}
logs/paddle-plsc/arcface/bz128/1n4g/r50_b128_fp32_5.log {1: 1555.18, 4: 1539.66, 2: 1534.62, 3: 1540.66, 5: 1535.04}
logs/paddle-plsc/arcface/bz128/1n1g/r50_b128_fp32_1.log {1: 397.44}
logs/paddle-plsc/arcface/bz128/1n1g/r50_b128_fp32_4.log {1: 397.44, 4: 397.64}
logs/paddle-plsc/arcface/bz128/1n1g/r50_b128_fp32_2.log {1: 397.44, 4: 397.64, 2: 398.05}
logs/paddle-plsc/arcface/bz128/1n1g/r50_b128_fp32_3.log {1: 397.44, 4: 397.64, 2: 398.05, 3: 398.98}
logs/paddle-plsc/arcface/bz128/1n1g/r50_b128_fp32_5.log {1: 397.44, 4: 397.64, 2: 398.05, 3: 398.98, 5: 397.78}
logs/paddle-plsc/arcface/bz128/2n8g/r50_b128_fp32_1.log {1: 5950.34}
logs/paddle-plsc/arcface/bz128/2n8g/r50_b128_fp32_4.log {1: 5950.34, 4: 5961.68}
logs/paddle-plsc/arcface/bz128/2n8g/r50_b128_fp32_2.log {1: 5950.34, 4: 5961.68, 2: 5953.84}
logs/paddle-plsc/arcface/bz128/2n8g/r50_b128_fp32_3.log {1: 5950.34, 4: 5961.68, 2: 5953.84, 3: 5982.75}
logs/paddle-plsc/arcface/bz128/2n8g/r50_b128_fp32_5.log {1: 5950.34, 4: 5961.68, 2: 5953.84, 3: 5982.75, 5: 5934.43}
{'r50': {'1n1g': {'average_speed': 397.98,
                  'batch_size_per_device': 128,
                  'median_speed': 397.78,
                  'speedup': 1.0},
         '1n4g': {'average_speed': 1541.03,
                  'batch_size_per_device': 128,
                  'median_speed': 1539.66,
                  'speedup': 3.87},
         '1n8g': {'average_speed': 2548.18,
                  'batch_size_per_device': 128,
                  'median_speed': 2545.3,
                  'speedup': 6.4},
         '2n8g': {'average_speed': 5956.61,
                  'batch_size_per_device': 128,
                  'median_speed': 5953.84,
                  'speedup': 14.97},
         '4n8g': {'average_speed': 11086.97,
                  'batch_size_per_device': 128,
                  'median_speed': 11084.53,
                  'speedup': 27.87}}}
Saving result to ./result/bz128_result.json
```

## 计算规则

### 1.测速脚本

- extract_paddle_logs.py
- extract_paddle_logs_time.py

两个脚本略有不同，得到的结果稍有误差：

extract_paddle_logs.py根据官方在log中打印的速度，在150个iter中，排除前50iter，取后100个iter的速度做平均；

extract_paddle_logs_time.py则根据log中打印出的时间，排除前50iter取后100个iter的实际运行时间计算速度。

README展示的是extract_paddle_logs.py的计算结果。

### 2.均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度

### 3.加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5



## ResNet 50 FP32


### batch size = 128

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 397.78    | 1       |
| 1        | 4       | 1539.66   | 3.87    |
| 1        | 8       | 2545.3    | 6.4     |
| 2        | 16      | 5953.84   | 14.97   |
| 4        | 32      | 11084.53  | 27.87   |




## 完整日志

[logs-20210312](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/PaddlePaddle/plsc/logs-20210312.zip)