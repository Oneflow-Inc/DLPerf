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
本次训练使用了ImageNet2012的一个子集(共54289张图片)，数据集制作以及格式参照[paddle官方说明](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification#%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)


# Training
集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里设置batch_size=128，从1机1卡～4机32卡进行了一组训练
## 单机
`models/PaddleCV/image_classification/`目录下,执行脚本
```shell
bash run_single_node.sh
```
对单机1卡、4卡、8卡分别做6组测试。
## 2机16卡
2机、4机等多机情况下，需要在所有机器节点上准备同样的数据集、执行同样的脚本，以完成分布式训练。


如，2机：NODE1='10.11.0.2'     NODE2='10.11.0.3' 的训练，需在两台机器上分别准备好数据集后，NODE1节点`models/PaddleCV/image_classification/`目录下,执行脚本:
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
[paddle.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/PaddlePaddle/paddle.zip)

## 加速比
执行以下脚本计算各个情况下的加速比：
```shell
python extract_paddle_logs.py --log_dir=./logs/resnet50
```
输出：
```shell
./logs/resnet50/4n8g/r50_b128_fp32_1.log {1: 9540.65}
./logs/resnet50/4n8g/r50_b128_fp32_4.log {1: 9540.65, 4: 9477.81}
./logs/resnet50/4n8g/r50_b128_fp32_2.log {1: 9540.65, 4: 9477.81, 2: 9423.41}
./logs/resnet50/4n8g/r50_b128_fp32_3.log {1: 9540.65, 4: 9477.81, 2: 9423.41, 3: 9458.98}
./logs/resnet50/4n8g/r50_b128_fp32_6.log {1: 9540.65, 4: 9477.81, 2: 9423.41, 3: 9458.98, 6: 9523.01}
./logs/resnet50/4n8g/r50_b128_fp32_5.log {1: 9540.65, 4: 9477.81, 2: 9423.41, 3: 9458.98, 6: 9523.01, 5: 9589.21}
./logs/resnet50/1n8g/r50_b128_fp32_1.log {1: 896.75}
./logs/resnet50/1n8g/r50_b128_fp32_4.log {1: 896.75, 4: 699.09}
./logs/resnet50/1n8g/r50_b128_fp32_2.log {1: 896.75, 4: 699.09, 2: 882.88}
./logs/resnet50/1n8g/r50_b128_fp32_3.log {1: 896.75, 4: 699.09, 2: 882.88, 3: 879.79}
./logs/resnet50/1n8g/r50_b128_fp32_6.log {1: 896.75, 4: 699.09, 2: 882.88, 3: 879.79, 6: 872.1}
./logs/resnet50/1n8g/r50_b128_fp32_5.log {1: 896.75, 4: 699.09, 2: 882.88, 3: 879.79, 6: 872.1, 5: 850.71}
./logs/resnet50/1n4g/r50_b128_fp32_1.log {1: 908.71}
./logs/resnet50/1n4g/r50_b128_fp32_4.log {1: 908.71, 4: 927.46}
./logs/resnet50/1n4g/r50_b128_fp32_2.log {1: 908.71, 4: 927.46, 2: 933.12}
./logs/resnet50/1n4g/r50_b128_fp32_3.log {1: 908.71, 4: 927.46, 2: 933.12, 3: 904.25}
./logs/resnet50/1n4g/r50_b128_fp32_6.log {1: 908.71, 4: 927.46, 2: 933.12, 3: 904.25, 6: 931.63}
./logs/resnet50/1n4g/r50_b128_fp32_5.log {1: 908.71, 4: 927.46, 2: 933.12, 3: 904.25, 6: 931.63, 5: 920.86}
./logs/resnet50/1n1g/r50_b128_fp32_1.log {1: 355.47}
./logs/resnet50/1n1g/r50_b128_fp32_4.log {1: 355.47, 4: 357.0}
./logs/resnet50/1n1g/r50_b128_fp32_2.log {1: 355.47, 4: 357.0, 2: 353.76}
./logs/resnet50/1n1g/r50_b128_fp32_3.log {1: 355.47, 4: 357.0, 2: 353.76, 3: 351.88}
./logs/resnet50/1n1g/r50_b128_fp32_6.log {1: 355.47, 4: 357.0, 2: 353.76, 3: 351.88, 6: 354.32}
./logs/resnet50/1n1g/r50_b128_fp32_5.log {1: 355.47, 4: 357.0, 2: 353.76, 3: 351.88, 6: 354.32, 5: 356.53}
./logs/resnet50/1n2g/r50_b128_fp32_1.log {1: 683.34}
./logs/resnet50/1n2g/r50_b128_fp32_4.log {1: 683.34, 4: 681.0}
./logs/resnet50/1n2g/r50_b128_fp32_2.log {1: 683.34, 4: 681.0, 2: 680.11}
./logs/resnet50/1n2g/r50_b128_fp32_3.log {1: 683.34, 4: 681.0, 2: 680.11, 3: 680.17}
./logs/resnet50/1n2g/r50_b128_fp32_6.log {1: 683.34, 4: 681.0, 2: 680.11, 3: 680.17, 6: 684.88}
./logs/resnet50/1n2g/r50_b128_fp32_5.log {1: 683.34, 4: 681.0, 2: 680.11, 3: 680.17, 6: 684.88, 5: 669.62}
./logs/resnet50/2n8g/r50_b128_fp32_1.log {1: 4754.98}
./logs/resnet50/2n8g/r50_b128_fp32_4.log {1: 4754.98, 4: 4763.01}
./logs/resnet50/2n8g/r50_b128_fp32_2.log {1: 4754.98, 4: 4763.01, 2: 4769.44}
./logs/resnet50/2n8g/r50_b128_fp32_3.log {1: 4754.98, 4: 4763.01, 2: 4769.44, 3: 4809.48}
./logs/resnet50/2n8g/r50_b128_fp32_6.log {1: 4754.98, 4: 4763.01, 2: 4769.44, 3: 4809.48, 6: 4800.01}
./logs/resnet50/2n8g/r50_b128_fp32_5.log {1: 4754.98, 4: 4763.01, 2: 4769.44, 3: 4809.48, 6: 4800.01, 5: 4788.84}
{'r50': {'1n1g': {'average_speed': 354.83,
                  'batch_size_per_device': 128,
                  'speedup': 1.0},
         '1n2g': {'average_speed': 679.85,
                  'batch_size_per_device': 128,
                  'speedup': 1.92},
         '1n4g': {'average_speed': 921.0,
                  'batch_size_per_device': 128,
                  'speedup': 2.58},
         '1n8g': {'average_speed': 846.89,
                  'batch_size_per_device': 128,
                  'speedup': 2.38},
         '2n8g': {'average_speed': 4780.96,
                  'batch_size_per_device': 128,
                  'speedup': 13.47},
         '4n8g': {'average_speed': 9502.18,
                  'batch_size_per_device': 128,
                  'speedup': 26.65}}}
```
## ResNet50 V1.5 bsz = 128

### FP32 & Without XLA

| 节点数 | GPU数 | samples/s(OneFlow) | 加速比 | samples/s(Paddle) | 加速比 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 383.76 | 1 | 354.83 | 1 |
| 1 | 4 | 1497.62 | 3.90 | 921.0 | 2.58 |
| 1 | 8 | 2942.32 | 7.67 | 846.89 | 2.38 |
| 2 | 16 | 5839.05 | 15.22 | 4780.96 | 13.47 |
| 4 | 32 | 11548.45 | 30.09 | 9502.18 | 26.65 |

附：[Paddle官方fp16+dali测试结果](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification#%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83)




