# 【DLPerf】NVIDIA-Tensorflow-BERT测评

# Overview
本次复现采用了[NVIDIA官方仓库](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590)中Tensorflow版[BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT)的实现，复现的目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。


- **Environment** 给出了测评时的硬件系统环境、Docker容器等信息
- **Quick Start** 介绍了从克隆官方Github仓库到数据集准备的详细过程
- **Training** 提供了方便易用的测评脚本，覆盖从单机单卡～多机多卡的情形
- **Result** 提供完整测评log日志，并给出示例代码，用以计算平均速度、加速比，并给出汇总表格



# Environment
## 系统

- 系统：Ubuntu 16.04
- 显卡：Tesla V100（16G）×8
- 驱动：440.33.01
- CUDA：10.2
- cuDNN：7.6.5
## NGC容器

- Ubuntu18.04
- Python 3.6
- **Tensorflow 1.15.2**
- CUDA 10.2.89
- cuDNN 7.6.5
- NCCL 2.6.3
- Horovod 0.19.0
- OpenMPI 3.1.4
- DALI 0.19.0
## Feature support matrix
| Feature | ResNet-50 v1.5 Tensorflow |
| --- | --- |
| Horovod Multi-GPU | Yes |
| Horovod Multi-Node | Yes |

# Quick Start
## 项目代码

- [NVIDIA官方仓库](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590)
   - [BERT项目主页](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT)

下载官方源码：
```shell
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples && checkout fed7ba99cde958fda12c9e81d12b3d7e738e0590
cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT/
```

1.将本页面scripts文件夹中的脚本：`run_pretraining_adam.sh`、`multi_node_run_pretraining_adam.sh`放入BERT/scripts下；

2.将scripts中的其余脚本：`SINGLE_NODE_BERT_FP32_1E.sh`、`TWO_NODE_BERT_FP32_1E.sh`和`MULTI_NODE_BERT_FP32_1E.sh`放入BERT目录下


## NGC容器
参考[NVIDIA官方Quick start](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT#quick-start-guide)
**构建项目镜像**

> 如果您本地有[nvidia:tensorflow](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)的NGC镜像，或者之前通过`docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3`下载过此镜像，则可以修改Dockerfile：
> 这将使得构建项目镜像时，直接从本地已有镜像构建（而不是从网上拉取）。

```shell
# 注释 ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.03-tf1-py3
ARG FROM_IMAGE_NAME=fdc4e72f4c15
```


本次测评采用的是NVIDIA官方提供的NGC镜像，您可以在目录DeepLearningExamples/TensorFlow/LanguageModeling/BERT/下运行
```shell
docker build . -t nvidia_bert_tf:20.03
```
直接构建本地项目镜像，Dockerfile中将通过`docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3`从网上拉取NVIDIA官方的NGC镜像。


**启动容器**
在DeepLearningExamples/TensorFlow/LanguageModeling/BERT/下运行：`bash scripts/docker/launch.sh` 以根据构建好的项目镜像启动容器，可以修改launch.sh为容器提供必要的启动参数：
```shell
# 启动容器
#!/bin/bash

CMD=${@:-/bin/bash}
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

nvidia-docker run  -it \
    --net=host --shm-size=16g \
    --ulimit memlock=-1 --privileged \
    --name tf_bert \
    --ulimit stack=67108864 \
    --device=/dev/infiniband \
    -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
    -v $PWD:/workspace/bert \
    -v /home/leinao/DLPerf/dataset/wiki/tfrecord:/workspace/bert/data/tfrecord \
    -v $PWD/results:/results \
    nvidia_bert_tf:20.03 $CMD
```
## 数据集
数据集准备见：[nvidia官方仓库说明](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT#default-configuration)
## SSH配置(可选)
单机情况下无需配置ssh服务，需要测试2机、4机等多机情况下，则需要安装docker容器间的ssh服务，配置ssh免密登录，保证分布式horovod/mpi脚本运行时可以在单机上与其他节点互联
**安装ssh服务端**
```shell
docker exec -it tf_resnet_lyon /bin/bash
apt-get update
apt-get install openssh-server
```
**设置免密登录**
1.容器生成ssh公钥
```shell
ssh-keygen -t rsa -C "xxxxx@xxxxx.com"
cat /root/.ssh/id_rsa.pub  >>   /root/.ssh/authorized_keys
```
2.各个节点间：/root/.ssh/id_rsa.pub 互相放到/root/.ssh/authorized_keys中
3.修改sshd中用于docker通信的Port端口号，以及相应配置：
`vim /etc/ssh/sshd_config`

```shell
Port 10000
#AddressFamily any
#ListenAddress 0.0.0.0
#ListenAddress ::

HostKey /root/.ssh/id_rsa
#HostKey /etc/ssh/ssh_host_rsa_key
#HostKey /etc/ssh/ssh_host_ecdsa_key
#HostKey /etc/ssh/ssh_host_ed25519_key

# Ciphers and keying
#RekeyLimit default none

# Logging
#SyslogFacility AUTH
#LogLevel INFO

# Authentication:

#LoginGraceTime 2m
PermitRootLogin yes
#PermitRootLogin prohibit-password
#StrictModes yes
#MaxAuthTries 6
#MaxSessions 10

PubkeyAuthentication yes

# Expect .ssh/authorized_keys2 to be disregarded by default in future.
AuthorizedKeysFile      .ssh/authorized_keys .ssh/authorized_keys2
```
4.重启ssh服务
`service ssh restart`
# Training
集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里设置batch_size=32，从1机1卡～4机32卡进行了一组训练
## 单机
进入容器：
```shell
docker exec -it nvidia_bert_tf /bin/bash
cd /workspace/bert
bash SINGLE_NODE_BERT_FP32_1E.sh
```
执行脚本，对单机1卡、2卡、4卡、8卡分别做6组测试。默认batch size为32，也可以通过参数指定batch size如：`bash SINGLE_NODE_BERT_FP32_1E.sh  48`
## 2机16卡
2机、4机等多机情况下，需要在所有机器节点上准备同样的数据集、以完成分布式训练。由于配置了ssh免密，您只需要在一个节点上运行脚本即可执行多机训练。

如2机：NODE1='10.11.0.2'   NODE2='10.11.0.3' 的训练，需在两台机器上分别准备好数据集后，NODE1节点进入容器/workspace/bert下，执行脚本:

`bash TWO_NODE_BERT_FP32_1E.sh`即可运行2机16卡的训练，同样默认测试6组。

## 4机32卡

流程同上，NODE1节点进入容器/workspace/bert目录下，执行脚本:

`bash MULTI_NODE_BERT_FP32_1E.sh`即可运行4机32卡的训练，测试6组。

## Result
### 完整日志
[log.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Tensorflow/bert/logs.zip)

### 加速比
执行以下脚本计算各个情况下的加速比：
```shell
python extract_tensorflow_logs.py --log_dir=logs/ngc/tensorflow/bert/bz48 --batch_size_per_device=48
```
输出：
```shell
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_2.log {2: 2482.36}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_1.log {2: 2482.36, 1: 2516.78}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_4.log {2: 2482.36, 1: 2516.78, 4: 2519.3}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_6.log {2: 2482.36, 1: 2516.78, 4: 2519.3, 6: 2580.03}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_5.log {2: 2482.36, 1: 2516.78, 4: 2519.3, 6: 2580.03, 5: 2262.68}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_3.log {2: 2482.36, 1: 2516.78, 4: 2519.3, 6: 2580.03, 5: 2262.68, 3: 2485.57}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_2.log {2: 853.73}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_1.log {2: 853.73, 1: 855.49}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_4.log {2: 853.73, 1: 855.49, 4: 852.75}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_6.log {2: 853.73, 1: 855.49, 4: 852.75, 6: 859.04}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_5.log {2: 853.73, 1: 855.49, 4: 852.75, 6: 859.04, 5: 855.64}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_3.log {2: 853.73, 1: 855.49, 4: 852.75, 6: 859.04, 5: 855.64, 3: 851.59}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_2.log {2: 430.6}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_1.log {2: 430.6, 1: 429.06}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_4.log {2: 430.6, 1: 429.06, 4: 430.36}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_6.log {2: 430.6, 1: 429.06, 4: 430.36, 6: 431.42}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_5.log {2: 430.6, 1: 429.06, 4: 430.36, 6: 431.42, 5: 430.16}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_3.log {2: 430.6, 1: 429.06, 4: 430.36, 6: 431.42, 5: 430.16, 3: 431.56}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_1.log {1: 112.81}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_4.log {1: 112.81, 4: 112.84}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_6.log {1: 112.81, 4: 112.84, 6: 112.62}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_5.log {1: 112.81, 4: 112.84, 6: 112.62, 5: 112.81}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_3.log {1: 112.81, 4: 112.84, 6: 112.62, 5: 112.81, 3: 112.6}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_2.log {2: 215.03}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_1.log {2: 215.03, 1: 214.4}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_4.log {2: 215.03, 1: 214.4, 4: 214.99}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_6.log {2: 215.03, 1: 214.4, 4: 214.99, 6: 214.76}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_5.log {2: 215.03, 1: 214.4, 4: 214.99, 6: 214.76, 5: 214.7}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_3.log {2: 215.03, 1: 214.4, 4: 214.99, 6: 214.76, 5: 214.7, 3: 215.03}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_2.log {2: 1408.27}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_1.log {2: 1408.27, 1: 1285.65}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_4.log {2: 1408.27, 1: 1285.65, 4: 1390.11}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_6.log {2: 1408.27, 1: 1285.65, 4: 1390.11, 6: 1379.22}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_5.log {2: 1408.27, 1: 1285.65, 4: 1390.11, 6: 1379.22, 5: 1395.64}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_3.log {2: 1408.27, 1: 1285.65, 4: 1390.11, 6: 1379.22, 5: 1395.64, 3: 1348.14}
{'bert': {'1n1g': {'average_speed': 112.74,
                   'batch_size_per_device': 48,
                   'median_speed': 112.81,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 214.82,
                   'batch_size_per_device': 48,
                   'median_speed': 214.88,
                   'speedup': 1.9},
          '1n4g': {'average_speed': 430.53,
                   'batch_size_per_device': 48,
                   'median_speed': 430.48,
                   'speedup': 3.82},
          '1n8g': {'average_speed': 854.71,
                   'batch_size_per_device': 48,
                   'median_speed': 854.61,
                   'speedup': 7.58},
          '2n8g': {'average_speed': 1367.84,
                   'batch_size_per_device': 48,
                   'median_speed': 1384.66,
                   'speedup': 12.27},
          '4n8g': {'average_speed': 2474.45,
                   'batch_size_per_device': 48,
                   'median_speed': 2501.18,
                   'speedup': 22.17}}}
Saving result to ./result/bz48_result.json
```
> 注：logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_2.log 由于日志异常，故不参与计算

### 计算规则
#### 1.测速脚本

- extract_tensorflow_logs.py


extract_tensorflow_logs.py根据官方在log中打印的速度，在120个iter中，排除前20iter，取后100个iter的速度做平均
#### 2.均值速度和中值速度

- average_speed均值速度

- median_speed中值速度
每个batch size进行5~7次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度

#### 3.加速比以中值速度计算
脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:
单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5
### BERT-Base  batch size=48
#### FP32 & Without XLA
| 节点数 | GPU数 | samples/s(Tensorflow) | 加速比 |
| --- | --- | --- | --- |
| 1 | 1 | 112.81 | 1.00 |
| 1 | 8 | 854.61 | 7.58 |
| 2 | 16 | 1384.66 | 12.27 |
| 4 | 32 | 2501.18 | 22.17 |



### BERT-Base  batch size=32
#### FP32 & Without XLA
| 节点数 | GPU数 | samples/s(OneFlow) | 加速比 | samples/s(Tensorflow) | 加速比 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 145.2 | 1.00 | 107.06 | 1.00 |
| 1 | 8 | 1043.0 | 7.18 | 791.76 | 7.40 |
| 2 | 16 | 1890.3 | 13.02 | 1103.3 | 10.31 |
| 4 | 32 | 3715.1 | 25.59 | 1967.05 | 18.37 |



