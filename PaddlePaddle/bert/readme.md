# 【DLPerf】Paddle-BERT测评

# Overview
本次复现采用了[PaddlePaddle官方仓库](https://github.com/PaddlePaddle/models/tree/release/1.8)中的paddle版[BERT](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT)的实现，复现的目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。


- **Environment  **给出了测评时的硬件系统环境、软件版本等信息
- **Quick Start     **介绍了从克隆官方Github仓库到数据集准备的详细过程
- **Training           **提供了方便易用的测评脚本，覆盖从单机单卡～多机多卡的情形
- **Result               **提供完整测评log日志，并给出示例代码，用以计算平均速度、加速比，并给出汇总表格



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
| Feature | BERT Paddle |
| --- | --- |
| BERT-Base | Yes |
| NVIDIA NCCL | Yes |
| FP32 | Yes |

# Quick Start
## 项目代码

- [PaddlePaddle官方仓库](https://github.com/PaddlePaddle/models/tree/release/1.8)
   - [BERT项目主页](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT)



下载官方源码：
```shell
git clone https://github.com/PaddlePaddle/models/tree/release/1.8
cd models/PaddleNLP/pretrain_language_models/BERT
```


将本页面scripts文件夹中的脚本：`make_pretrain_data.sh` 放入BERT/data下，其余脚本全部放入：BERT/目录下
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
本次bert的预训练过程使用了paddle官方的示例数据集：[demo_wiki_train.gz](https://github.com/PaddlePaddle/models/blob/release/1.8/PaddleNLP/pretrain_language_models/BERT/data/train/demo_wiki_train.gz)，由于数据集规模较小，我们在此基础上制作了demo_wiki_train_50.gz用于预训练。数据集制作过程如下：

`cd models/PaddleNLP/pretrain_language_models/BERT/data`

`bash make_pretrain_data.sh`

脚本将复制demo_wiki_train的内容，构造出一个50倍数据规模的训练集demo_wiki_train_50.gz

# Training
集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5



每个节点有8张显卡，这里设置batch size为32、64和96，分别在1机1卡～4机32卡的情况下进行了多组训练。
## 单机
`models/PaddleNLP/pretrain_language_models/BERT`目录下,执行脚本
```shell
bash run_single_node.sh
```
对单机1卡、2卡、4卡、8卡分别做6组测试。单机多机脚本默认的batch size为32，可以通过参数指定，如指定batch size为64或96：`bash run_single_node.sh 64`，`bash run_single_node.sh 96`
## 2机16卡
2机、4机等多机情况下，需要在所有机器节点上准备同样的数据集、执行同样的脚本，以完成分布式训练。


如，2机：NODE1='10.11.0.2'     NODE2='10.11.0.3' 的训练，需在两台机器上分别准备好数据集后，NODE1节点`models/PaddleNLP/pretrain_language_models/BERT/`目录下,执行脚本:
```shell
bash run_two_node.sh
```
NODE2节点`models/PaddleNLP/pretrain_language_models/BERT/`目录下，执行同样的脚本`bash run_two_node.sh`，即可运行2机16卡的训练，同样默认测试6组。
## 4机32卡
流程同上，在4个机器节点上分别执行：`
`
```shell
bash run_multi_node.sh
```
以运行4机32卡的训练，默认测试6组。
# Result
## 完整日志
（TOTO）
## 加速比
执行以下脚本计算各个情况下的加速比：
```shell
python extract_paddle_logs.py --log_dir=./logs/paddle/bert
```
输出：（TOTO）
```shell

```
## BERT-Base  batch size=32
### FP32 & Without XLA
| 节点数 | GPU数 | samples/s(OneFlow) | 加速比 | samples/s(Paddle) | 加速比 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 145.2 | 1.00 |    |    |
| 1 | 4 |    |    |    |    |
| 1 | 8 | 1043.0 | 7.18 |    |    |
| 2 | 16 | 1890.3 | 13.02 |    |    |
| 4 | 32 | 3715.1 | 25.59 |    |    |

## BERT-Base  batch size=64
### FP32 & Without XLA
| 节点数 | GPU数 | samples/s(OneFlow) | 加速比 | samples/s(Paddle) | 加速比 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 149.8 | 1 |    |    |
| 1 | 4 |             |     |    |    |
| 1 | 8 | 1138.9 | 7.60 |    |    |
| 2 | 16 | 2189.3 | 14.61 |    |   |
| 4 | 32 | 4310.4 | 28.77 |    |    |

## BERT-Base  batch size=96
### FP32 & Without XLA
| 节点数 | GPU数 | samples/s(OneFlow) | 加速比 | samples/s(Paddle) | 加速比 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 149.8 | 1.00 |    |    |
| 1 | 4 |    |    |    |    |
| 1 | 8 | 1158.5 | 7.73 |    |    |
| 2 | 16 | 2257.7 | 15.07 |    |    |
| 4 | 32 | 4456.0 | 29.75 |    |    |



