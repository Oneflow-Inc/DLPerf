# 【DLPerf】Tensorflow-ResNet50V1.5测评

# Overview
本次复现采用了[Tensorflow官方仓库](https://github.com/tensorflow/models/tree/r2.3.0)中的tensorflow2.3版的[ResNet50(v1.5)](https://github.com/tensorflow/models/tree/r2.3.0/official/vision/image_classification)的实现，复现的目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。


- **Environment** 给出了测评时的硬件系统环境、软件版本等信息
- **Quick Start** 介绍了从克隆官方Github仓库到数据集准备的详细过程
- **Training** 提供了方便易用的测评脚本，覆盖从单机单卡～多机多卡的情形
- **Result** 提供完整测评log日志，并给出示例代码，用以计算平均速度、加速比，并给出汇总表格


# Environment
## 系统

- 系统：Ubuntu 16.04
- 显卡：Tesla V100（16G）×8
- 驱动：Nvidia 440.33.01
- CUDA：10.2
- cuDNN：7.6.5
- NCCL：2.7.3
## 框架

- **Tensorflow 2.3.0** 

  

# Quick Start
## 框架安装
```shell
python -m pip install tensorflow==2.3.0 -i https://mirror.baidu.com/pypi/simple
```
## 项目代码

- [Tensorflow官方仓库](https://github.com/tensorflow/models/tree/r2.3.0)
   - [Resnet50_v1.5项目主页](https://github.com/tensorflow/models/tree/r2.3.0/official/vision/image_classification)

下载官方源码：
```shell
git clone https://github.com/tensorflow/models.git && checkout r2.3.0
cd models/official/vision/image_classification
# 根据项目official目录下的 requirements.txt安装依赖(需要指定tensorflow-datasets==3.0.0，否则可能报错)
python -m pip install -r requirements.txt
```
将本页面scripts文件夹中的.yaml配置文件放入image_classification/configs/examples/resnet/imagenet/下，其余.sh脚本全部放入：image_classification/目录下。
## NCCL
tensorflow的分布式训练底层依赖nccl库，需要从[NVIDIA-NCCL官网下载](https://developer.nvidia.com/nccl/nccl-download)并安装和操作系统、cuda版本适配的nccl。例如：安装2.7.3版本的nccl：
```shell
sudo dpkg -i nccl-repo-ubuntu1604-2.7.3-ga-cuda10.2_1-1_amd64.deb
sudo apt update
sudo apt install libnccl2=2.7.3-1+cuda10.2 libnccl-dev=2.7.3-1+cuda10.2
```
## 数据集

本次训练使用了ImageNet2012的一个子集：train-00000-of-01024,train-00001-of-01024...train-0000511-of-01024共640512张图片，数据集格式为tfrecord，具体制作方法参考tensorflow官方。



# Training

集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里设置batch_size=128，从1机1卡～4机32卡进行了一组训练。
## 单机
`models/official/vision/image_classification`目录下,执行脚本
```shell
bash run_single_node.sh
```
对单机1卡、2卡、4卡、8卡分别做5组测试。
## 2机16卡
2机、4机等多机情况下，需要在所有机器节点上准备同样的数据集、执行同样的脚本，以完成分布式训练。如，2机：NODE1='10.11.0.2'     NODE2='10.11.0.3' 的训练，需在两台机器上分别准备好数据集后，NODE1节点`models/official/vision/image_classification/`目录下,执行脚本:

```shell
bash run_two_node.sh
```
NODE2节点相同目录下，执行同样的脚本`bash run_two_node.sh`即可运行2机16卡的训练，同样默认测试5组。
## 4机32卡
流程同上，在4个机器节点上分别执行：`
`
```shell
bash run_multi_node.sh
```
以运行4机32卡的训练，默认测试5组。
# Result
## 完整日志
[logs.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/Tensorflow/resnet50/logs.zip)
## 加速比
执行以下脚本计算各个情况下的加速比：
```shell
python extract_tensorflow_logs.py --log_dir=logs/tensorflow/resnet50
```
输出：
```shell
logs/tensorflow/resnet50/4n8g/rn50_b128_fp32_1.log {1: 9369.06}
logs/tensorflow/resnet50/4n8g/rn50_b128_fp32_3.log {1: 9369.06, 3: 9418.44}
logs/tensorflow/resnet50/4n8g/rn50_b128_fp32_5.log {1: 9369.06, 3: 9418.44, 5: 9454.31}
logs/tensorflow/resnet50/4n8g/rn50_b128_fp32_4.log {1: 9369.06, 3: 9418.44, 5: 9454.31, 4: 9420.23}
logs/tensorflow/resnet50/4n8g/rn50_b128_fp32_2.log {1: 9369.06, 3: 9418.44, 5: 9454.31, 4: 9420.23, 2: 9402.51}
logs/tensorflow/resnet50/1n8g/rn50_b128_fp32_1.log {1: 2494.22}
logs/tensorflow/resnet50/1n8g/rn50_b128_fp32_3.log {1: 2494.22, 3: 2458.74}
logs/tensorflow/resnet50/1n8g/rn50_b128_fp32_5.log {1: 2494.22, 3: 2458.74, 5: 2455.28}
logs/tensorflow/resnet50/1n8g/rn50_b128_fp32_4.log {1: 2494.22, 3: 2458.74, 5: 2455.28, 4: 2480.44}
logs/tensorflow/resnet50/1n8g/rn50_b128_fp32_2.log {1: 2494.22, 3: 2458.74, 5: 2455.28, 4: 2480.44, 2: 2441.07}
logs/tensorflow/resnet50/1n4g/rn50_b128_fp32_1.log {1: 1145.76}
logs/tensorflow/resnet50/1n4g/rn50_b128_fp32_3.log {1: 1145.76, 3: 1167.06}
logs/tensorflow/resnet50/1n4g/rn50_b128_fp32_5.log {1: 1145.76, 3: 1167.06, 5: 1128.34}
logs/tensorflow/resnet50/1n4g/rn50_b128_fp32_4.log {1: 1145.76, 3: 1167.06, 5: 1128.34, 4: 1122.51}
logs/tensorflow/resnet50/1n4g/rn50_b128_fp32_2.log {1: 1145.76, 3: 1167.06, 5: 1128.34, 4: 1122.51, 2: 1143.46}
logs/tensorflow/resnet50/1n1g/rn50_b128_fp32_1.log {1: 323.9}
logs/tensorflow/resnet50/1n1g/rn50_b128_fp32_3.log {1: 323.9, 3: 320.18}
logs/tensorflow/resnet50/1n1g/rn50_b128_fp32_5.log {1: 323.9, 3: 320.18, 5: 321.22}
logs/tensorflow/resnet50/1n1g/rn50_b128_fp32_4.log {1: 323.9, 3: 320.18, 5: 321.22, 4: 322.25}
logs/tensorflow/resnet50/1n1g/rn50_b128_fp32_2.log {1: 323.9, 3: 320.18, 5: 321.22, 4: 322.25, 2: 321.8}
logs/tensorflow/resnet50/1n2g/rn50_b128_fp32_1.log {1: 575.76}
logs/tensorflow/resnet50/1n2g/rn50_b128_fp32_3.log {1: 575.76, 3: 574.66}
logs/tensorflow/resnet50/1n2g/rn50_b128_fp32_5.log {1: 575.76, 3: 574.66, 5: 563.93}
logs/tensorflow/resnet50/1n2g/rn50_b128_fp32_4.log {1: 575.76, 3: 574.66, 5: 563.93, 4: 571.71}
logs/tensorflow/resnet50/1n2g/rn50_b128_fp32_2.log {1: 575.76, 3: 574.66, 5: 563.93, 4: 571.71, 2: 580.55}
logs/tensorflow/resnet50/2n8g/rn50_b128_fp32_1.log {1: 4864.58}
logs/tensorflow/resnet50/2n8g/rn50_b128_fp32_3.log {1: 4864.58, 3: 4829.93}
logs/tensorflow/resnet50/2n8g/rn50_b128_fp32_5.log {1: 4864.58, 3: 4829.93, 5: 4789.87}
logs/tensorflow/resnet50/2n8g/rn50_b128_fp32_4.log {1: 4864.58, 3: 4829.93, 5: 4789.87, 4: 4849.68}
logs/tensorflow/resnet50/2n8g/rn50_b128_fp32_2.log {1: 4864.58, 3: 4829.93, 5: 4789.87, 4: 4849.68, 2: 4918.45}
{'rn50': {'1n1g': {'average_speed': 321.87,
                   'batch_size_per_device': 128,
                   'median_speed': 321.8,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 573.32,
                   'batch_size_per_device': 128,
                   'median_speed': 574.66,
                   'speedup': 1.79},
          '1n4g': {'average_speed': 1141.43,
                   'batch_size_per_device': 128,
                   'median_speed': 1143.46,
                   'speedup': 3.55},
          '1n8g': {'average_speed': 2465.95,
                   'batch_size_per_device': 128,
                   'median_speed': 2458.74,
                   'speedup': 7.64},
          '2n8g': {'average_speed': 4850.5,
                   'batch_size_per_device': 128,
                   'median_speed': 4849.68,
                   'speedup': 15.07},
          '4n8g': {'average_speed': 9412.91,
                   'batch_size_per_device': 128,
                   'median_speed': 9418.44,
                   'speedup': 29.27}}}
Saving result to ./result/resnet50_result.json
```
## 计算规则
### 1.测速脚本

- extract_tensorflow_logs.py
- extract_tensorflow_logs_time.py

两个脚本略有不同：

extract_tensorflow_logs.py根据官方在log中打印的速度，在600个iter中，排除前100iter，取后500个iter的速度做平均；

extract_tensorflow_logs_time.py则排除前100iter，取后500个iter的实际运行时间计算速度。

### 2.均值速度和中值速度

- average_speed均值速度

- median_speed中值速度每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度
### 3.加速比以中值速度计算
脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5
## ResNet50 V1.5 batch szie = 128
### FP32 & Without XLA
| 节点数 | GPU数 | samples/s | 加速比 |
| --- | --- | --- | --- |
| 1 | 1 | 321.8 | 1 |
| 1 | 2 | 574.66 | 1.79 |
| 1 | 4 | 1143.46 | 3.55 |
| 1 | 8 | 2458.74 | 7.64 |
| 2 | 16 | 4849.68 | 15.07 |
| 4 | 32 | 9418.44 | 29.27 |



