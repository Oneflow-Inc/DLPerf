# 【DLPerf】Paddle-ResNet50V1.5测评

# Overview
本次复现采用了[PaddlePaddle官方仓库](https://github.com/PaddlePaddle/models/tree/release/1.8)中的paddle版[ResNet50(v1.5)](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification)的实现，复现的目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，我们仅测试了正常FP32精度下，不加XLA时的情况，后续我们会陆续开展混合精度、XLA等多种方式的测评。



本文主要内容：


- **Environment**

  - 给出了测评时的硬件系统环境、软件版本等信息
- **Quick Start**

  - 介绍了从克隆官方Github仓库到数据集准备的详细过程
- **Training**

  - 提供了方便易用的测评脚本，覆盖从单机单卡～多机多卡的情形
- **Result**

  - 提供完整测评log日志，并给出示例代码，用以计算平均速度、加速比，并给出汇总表格



# Environment
## 系统

- 系统：Ubuntu 16.04
- 显卡：Tesla V100（16G）×8
- 驱动：Nvidia 440.33.01
- cuda：10.2
- cudnn：7.6.5
- nccl：2.7.3
## 框架

- **paddle 1.8.3.post107**
## Feature support matrix
| Feature | ResNet-50 v1.5 Paddle |
| --- | --- |
| Multi-node,multi-gpu training | Yes |
| NVIDIA NCCL | Yes |

# Quick Start
## 项目代码

- [PaddlePaddle官方仓库](https://github.com/PaddlePaddle/models/tree/release/1.8)
   - [Resnet50_v1.5项目主页](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification)

下载官方源码：
```shell
git clone https://github.com/PaddlePaddle/models/tree/release/1.8
cd models/PaddleCV/image_classification
```


将本页面scripts文件夹中的脚本全部放入：`models/PaddleCV/image_classification/`目录下
## 框架安装
```shell
python3 -m pip install paddlepaddle-gpu==1.8.3.post107 -i https://mirror.baidu.com/pypi/simple
```
## NCCL
paddle的分布式训练底层依赖nccl库，需要从[NVIDIA-NCCL官网下载](https://developer.nvidia.com/nccl/nccl-download)并安装和操作系统、cuda版本适配的nccl。
例如：安装2.7.3版本的nccl：
```shell
sudo dpkg -i nccl-repo-ubuntu1604-2.7.3-ga-cuda10.2_1-1_amd64.deb
sudo apt update
sudo apt install libnccl2=2.7.3-1+cuda10.2 libnccl-dev=2.7.3-1+cuda10.2
```
## 数据集
本次训练使用了ImageNet2012的一个子集(共651468张图片)，数据集制作以及格式参照[paddle官方说明](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification#%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)


# Training
集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

## 测评方法

每个节点有8张显卡，这里设置batch_size=128，从1机1卡～4机32卡进行了多组训练。每组进行5~7次训练，每次训练过程只取第1个epoch的前120iter，计算训练速度时去掉前20iter，只取后100iter。最后将5~7次训练的速度求平均得到平均速度。

在下文的【Result】部分，我们提供了完整日志计算平均速度的完整代码。

## 单机
`models/PaddleCV/image_classification/`目录下,执行脚本
```shell
bash run_single_node.sh
```
对单机1卡、4卡、8卡分别做6组测试。
## 2机16卡
2机、4机等多机情况下，需要在所有机器节点上准备同样的数据集、执行同样的脚本，以完成分布式训练。如，2机：NODE1='10.11.0.2'     NODE2='10.11.0.3' 的训练，需在两台机器上分别准备好数据集后，NODE1节点`models/PaddleCV/image_classification/`目录下,执行脚本:

```shell
bash run_two_node.sh
```
NODE2节点`models/PaddleCV/image_classification/`目录下，执行同样的脚本`bash run_two_node.sh`
即可运行2机16卡的训练，同样默认测试6组。
## 4机32卡
流程同上，在4个机器节点上分别执行：`
`
```shell
bash run_multi_node.sh
```
以运行4机32卡的训练，默认测试6组。
# Result
## 完整日志
[resnet50.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/PaddlePaddle/resnet50.zip)

## 加速比
这里提供了两个脚本

执行以下脚本计算各个情况下的加速比：

```shell
python extract_paddle_logs_time.py  --log_dir=logs/paddle/resnet50
```
输出：
```shell
logs/paddle/resnet50/4n8g/r50_b128_fp32_1.log {1: 9337.1}
logs/paddle/resnet50/4n8g/r50_b128_fp32_4.log {1: 9337.1, 4: 9427.36}
logs/paddle/resnet50/4n8g/r50_b128_fp32_2.log {1: 9337.1, 4: 9427.36, 2: 9423.89}
logs/paddle/resnet50/4n8g/r50_b128_fp32_3.log {1: 9337.1, 4: 9427.36, 2: 9423.89, 3: 9431.7}
logs/paddle/resnet50/4n8g/r50_b128_fp32_6.log {1: 9337.1, 4: 9427.36, 2: 9423.89, 3: 9431.7, 6: 9388.9}
logs/paddle/resnet50/4n8g/r50_b128_fp32_5.log {1: 9337.1, 4: 9427.36, 2: 9423.89, 3: 9431.7, 6: 9388.9, 5: 9426.49}
logs/paddle/resnet50/1n8g/r50_b128_fp32_1.log {1: 2634.08}
logs/paddle/resnet50/1n8g/r50_b128_fp32_4.log {1: 2634.08, 4: 2659.53}
logs/paddle/resnet50/1n8g/r50_b128_fp32_2.log {1: 2634.08, 4: 2659.53, 2: 2647.16}
logs/paddle/resnet50/1n8g/r50_b128_fp32_3.log {1: 2634.08, 4: 2659.53, 2: 2647.16, 3: 2644.29}
logs/paddle/resnet50/1n8g/r50_b128_fp32_6.log {1: 2634.08, 4: 2659.53, 2: 2647.16, 3: 2644.29, 6: 2655.33}
logs/paddle/resnet50/1n8g/r50_b128_fp32_5.log {1: 2634.08, 4: 2659.53, 2: 2647.16, 3: 2644.29, 6: 2655.33, 5: 2645.65}
logs/paddle/resnet50/1n4g/r50_b128_fp32_1.log {1: 1365.7}
logs/paddle/resnet50/1n4g/r50_b128_fp32_4.log {1: 1365.7, 4: 1366.86}
logs/paddle/resnet50/1n4g/r50_b128_fp32_2.log {1: 1365.7, 4: 1366.86, 2: 1374.94}
logs/paddle/resnet50/1n4g/r50_b128_fp32_3.log {1: 1365.7, 4: 1366.86, 2: 1374.94, 3: 1376.38}
logs/paddle/resnet50/1n4g/r50_b128_fp32_6.log {1: 1365.7, 4: 1366.86, 2: 1374.94, 3: 1376.38, 6: 1370.67}
logs/paddle/resnet50/1n4g/r50_b128_fp32_5.log {1: 1365.7, 4: 1366.86, 2: 1374.94, 3: 1376.38, 6: 1370.67, 5: 1372.4}
logs/paddle/resnet50/1n1g/r50_b128_fp32_1.log {1: 357.34}
logs/paddle/resnet50/1n1g/r50_b128_fp32_4.log {1: 357.34, 4: 354.08}
logs/paddle/resnet50/1n1g/r50_b128_fp32_2.log {1: 357.34, 4: 354.08, 2: 357.86}
logs/paddle/resnet50/1n1g/r50_b128_fp32_3.log {1: 357.34, 4: 354.08, 2: 357.86, 3: 356.06}
logs/paddle/resnet50/1n1g/r50_b128_fp32_6.log {1: 357.34, 4: 354.08, 2: 357.86, 3: 356.06, 6: 355.1}
logs/paddle/resnet50/1n1g/r50_b128_fp32_5.log {1: 357.34, 4: 354.08, 2: 357.86, 3: 356.06, 6: 355.1, 5: 354.32}
logs/paddle/resnet50/1n2g/r50_b128_fp32_1.log {1: 661.52}
logs/paddle/resnet50/1n2g/r50_b128_fp32_4.log {1: 661.52, 4: 632.27}
logs/paddle/resnet50/1n2g/r50_b128_fp32_2.log {1: 661.52, 4: 632.27, 2: 637.67}
logs/paddle/resnet50/1n2g/r50_b128_fp32_3.log {1: 661.52, 4: 632.27, 2: 637.67, 3: 645.26}
logs/paddle/resnet50/1n2g/r50_b128_fp32_6.log {1: 661.52, 4: 632.27, 2: 637.67, 3: 645.26, 6: 617.69}
logs/paddle/resnet50/1n2g/r50_b128_fp32_5.log {1: 661.52, 4: 632.27, 2: 637.67, 3: 645.26, 6: 617.69, 5: 645.89}
logs/paddle/resnet50/2n8g/r50_b128_fp32_1.log {1: 4934.46}
logs/paddle/resnet50/2n8g/r50_b128_fp32_4.log {1: 4934.46, 4: 4937.68}
logs/paddle/resnet50/2n8g/r50_b128_fp32_2.log {1: 4934.46, 4: 4937.68, 2: 4946.86}
logs/paddle/resnet50/2n8g/r50_b128_fp32_3.log {1: 4934.46, 4: 4937.68, 2: 4946.86, 3: 4962.8}
logs/paddle/resnet50/2n8g/r50_b128_fp32_6.log {1: 4934.46, 4: 4937.68, 2: 4946.86, 3: 4962.8, 6: 4936.37}
logs/paddle/resnet50/2n8g/r50_b128_fp32_5.log {1: 4934.46, 4: 4937.68, 2: 4946.86, 3: 4962.8, 6: 4936.37, 5: 4920.83}
{'r50': {'1n1g': {'average_speed': 355.79,
                  'batch_size_per_device': 128,
                  'speedup': 1.0},
         '1n2g': {'average_speed': 640.05,
                  'batch_size_per_device': 128,
                  'speedup': 1.8},
         '1n4g': {'average_speed': 1371.16,
                  'batch_size_per_device': 128,
                  'speedup': 3.87},
         '1n8g': {'average_speed': 2647.67,
                  'batch_size_per_device': 128,
                  'speedup': 7.47},
         '2n8g': {'average_speed': 4939.83,
                  'batch_size_per_device': 128,
                  'speedup': 13.88},
         '4n8g': {'average_speed': 9405.91,
                  'batch_size_per_device': 128,
                  'speedup': 26.55}}}
Saving result to ./result/resnet50_result.json
```
## ResNet50 V1.5 bsz = 128

### FP32 & Without XLA

| 节点数 | GPU数 | samples/s(OneFlow) | 加速比 | samples/s(Paddle) | 加速比 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 383.76 | 1 | 355.79            | 1 |
| 1 | 4 | 1497.62 | 3.90 | 1371.16           | 3.87   |
| 1 | 8 | 2942.32 | 7.67 | 2647.67           | 7.47   |
| 2 | 16 | 5839.05 | 15.22 | 4939.83           | 13.88  |
| 4 | 32 | 11548.45 | 30.09 | 9405.91 | 26.55 |

附：[Paddle官方fp16+dali测试结果](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification#%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83)



