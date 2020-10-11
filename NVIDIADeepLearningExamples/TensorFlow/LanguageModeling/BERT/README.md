# 【DLPerf】NVIDIA-TensorFlow-BERT测评

# Overview

本次复现采用了[NVIDIA官方仓库](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590)中TensorFlow版[BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT)，目的在于速度测评，同时根据测速结果给出1机、2机、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试已覆盖 FP32、FP16混合精度以及XLA，后续将持续维护，增加更多方式的测评。



# Environment

## 系统

- 系统：Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- 显卡：Tesla V100-SXM2-16GB x 8
- 驱动：NVIDIA 440.33.01
- CUDA：10.2
- cuDNN：7.6.5

## NGC容器

- Ubuntu18.04
- Python 3.6
- **TensorFlow 1.15.2**
- CUDA 10.2.89
- cuDNN 7.6.5
- NCCL 2.6.3
- Horovod 0.19.0
- OpenMPI 3.1.4
- DALI 0.19.0

## Feature support matrix

| Feature            | ResNet-50 v1.5 TensorFlow |
| ------------------ | ------------------------- |
| Horovod Multi-gpu | Yes                       |
| Horovod Multi-node | Yes                       |
| Automatic mixed precision (AMP) | Yes                       |

# Quick Start

## 项目代码

- [NVIDIA官方仓库](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590)
  - [BERT项目主页](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT)

下载官方源码：

```shell
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples && git checkout fed7ba99cde958fda12c9e81d12b3d7e738e0590
cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT/
```

1.将本页面scripts路径下的脚本：`run_pretraining_adam.sh`、`multi_node_run_pretraining_adam.sh`放入BERT/scripts下；

2.将scripts路径下的其余脚本：`SINGLE_NODE_BERT_FP32_1E.sh`、`TWO_NODE_BERT_FP32_1E.sh`和`MULTI_NODE_BERT_FP32_1E.sh`放入BERT路径下


## NGC容器

参考[NVIDIA官方Quick start](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT#quick-start-guide)
**构建项目镜像**

本次测评采用的是NVIDIA官方提供的NGC镜像，您可以在目录DeepLearningExamples/TensorFlow/LanguageModeling/BERT/下运行

```shell
docker build . -t nvidia_bert_tf:20.03
```

直接构建本地项目镜像，Dockerfile中将通过`docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3`从网上拉取NVIDIA官方的NGC镜像。

> 如果您之前通过`docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3`下载过此镜像，或者本地有[nvidia:tensorflow](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)的NGC镜像，则可以修改Dockerfile：
>
> ```
> # 注释 ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.03-tf1-py3
> ARG FROM_IMAGE_NAME=fdc4e72f4c15
> ```
>
> 这将使得构建项目镜像时，直接从本地已有镜像构建（而不是从网上拉取）。

**启动容器**
根据构建好的项目镜像启动容器，在DeepLearningExamples/TensorFlow/LanguageModeling/BERT/下运行：`bash scripts/docker/launch.sh` ，修改launch.sh，为容器提供必要的启动参数：

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

数据集准备详见：[nvidia官方仓库说明](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT#default-configuration)

## SSH配置(可选)

单机情况下无需配置ssh服务，需要测试2机、4机等情况下时，则需要安装docker容器间的ssh服务，配置ssh免密登录，保证分布式horovod/mpi脚本运行时可以在多机间互联。
**安装ssh服务端**

```shell
docker exec -it tf_resnet  /bin/bash
apt-get update
apt-get install openssh-server
```

**设置免密登录**

- 1.节点间的 /root/.ssh/id_rsa.pub 互相授权，添加到 /root/.ssh/authorized_keys 中
- 2.修改sshd中用于docker通信的Port端口号，以及相应配置：`vim /etc/ssh/sshd_config`

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
...
```

- 3.重启ssh服务`service ssh restart`

## IB驱动安装（可选）

如果服务器之间支持IB(**InfiniBand**)网络，则可以安装IB驱动，使得多机情况下各个节点间的通信速率明显提升，从而加速框架在多机环境下的训练，提升加速比。

```shell
apt-get update
apt install dpatch libelf1 libmnl0 libltdl-dev lsof chrpath debhelper pciutils tk bison graphviz ethtool kmod gfortran swig flex tcl
```

从[NVIDIA官网](https://www.mellanox.com/products/InfiniBand-VPI-Software)下载适合操作系统及相应版本的IB驱动包，如果是nvidia-ngc容器，可以直接使用我们提高好的驱动包：下载[IB驱动 MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64.tar 源码包](http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64.tar)并解压

```
wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64.tar && tar -xvf MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64.tar
```

进入源码包路径，安装

```
cd MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64 && ./mlnxofedinstall --user-space-only --without-fw-update --all --force 
```

完成后，可以通过`ibstat`命令检查驱动是否安装成功。

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
bash run_single_node.sh
```

执行脚本，对单机1卡、2卡、4卡、8卡分别做5次测试（默认测试fp32+batch size32）。也可以通过参数指定batch size，如：`bash run_single_node.sh  48`

### 混合精度&XLA

可以修改脚本中的变量，或直接指定运行参数，例如：

- batch size=64 使用fp16混合精度：

```shell
bash   run_single_node.sh    64   fp16
```

- batch size=64 使用fp16混合精度 + XLA：

```shell
bash   run_single_node.sh    64   fp16     true
```

## 2机16卡

2机、4机等多机情况下，需要在所有机器节点上相同路径准备同样的数据集、以完成分布式训练。由于配置了ssh免密，您只需要在一个节点上运行脚本即可执行多机训练。

如2机：NODE1='10.11.0.2'   NODE2='10.11.0.3' 的训练，需在两台机器上分别准备好数据集后，NODE1节点进入容器/workspace/bert下，执行脚本:

`bash run_two_node.sh`即可运行2机16卡的训练，同样默认测试5次。

### 混合精度&XLA

可以修改脚本中的变量，或直接指定运行参数，例如：

- batch size=64 使用fp16混合精度 + XLA：

```shell
bash   run_two_node.sh    64   fp16    true
```

## 4机32卡

流程同上，NODE1节点进入容器/workspace/bert目录下，执行脚本:

`bash run_multi_node.sh`即可运行4机32卡的训练，测试5次。

### 混合精度&XLA

可以修改脚本中的变量，或直接指定运行参数，例如：

- batch size=64 使用fp16混合精度 + XLA：

```shell
bash   run_multi_node.sh    64   fp16    true
```

## Result

### 吞吐率及加速比

执行以下命令，即可计算各种测试配置下的吞吐率及加速比：

```shell
python extract_tensorflow_logs_time.py --log_dir=logs/ngc/tensorflow/bert/bz48 --batch_size_per_device=48
```

输出：

```shell
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_2.log {2: 2493.48}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_1.log {2: 2493.48, 1: 2157.02}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_4.log {2: 2493.48, 1: 2157.02, 4: 2463.7}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_6.log {2: 2493.48, 1: 2157.02, 4: 2463.7, 6: 2501.03}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_5.log {2: 2493.48, 1: 2157.02, 4: 2463.7, 6: 2501.03, 5: 2600.51}
logs/ngc/tensorflow/bert/bz48/4n8g/bert_b48_fp32_3.log {2: 2493.48, 1: 2157.02, 4: 2463.7, 6: 2501.03, 5: 2600.51, 3: 2208.05}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_2.log {2: 865.89}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_1.log {2: 865.89, 1: 869.25}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_4.log {2: 865.89, 1: 869.25, 4: 867.21}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_6.log {2: 865.89, 1: 869.25, 4: 867.21, 6: 866.16}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_5.log {2: 865.89, 1: 869.25, 4: 867.21, 6: 866.16, 5: 868.23}
logs/ngc/tensorflow/bert/bz48/1n8g/bert_b48_fp32_3.log {2: 865.89, 1: 869.25, 4: 867.21, 6: 866.16, 5: 868.23, 3: 869.02}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_2.log {2: 437.16}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_1.log {2: 437.16, 1: 436.61}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_4.log {2: 437.16, 1: 436.61, 4: 436.23}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_6.log {2: 437.16, 1: 436.61, 4: 436.23, 6: 437.6}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_5.log {2: 437.16, 1: 436.61, 4: 436.23, 6: 437.6, 5: 436.78}
logs/ngc/tensorflow/bert/bz48/1n4g/bert_b48_fp32_3.log {2: 437.16, 1: 436.61, 4: 436.23, 6: 437.6, 5: 436.78, 3: 436.38}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_2.log {2: 112.72}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_1.log {2: 112.72, 1: 113.4}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_4.log {2: 112.72, 1: 113.4, 4: 112.48}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_6.log {2: 112.72, 1: 113.4, 4: 112.48, 6: 112.34}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_5.log {2: 112.72, 1: 113.4, 4: 112.48, 6: 112.34, 5: 112.5}
logs/ngc/tensorflow/bert/bz48/1n1g/bert_b48_fp32_3.log {2: 112.72, 1: 113.4, 4: 112.48, 6: 112.34, 5: 112.5, 3: 112.62}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_2.log {2: 217.4}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_1.log {2: 217.4, 1: 217.34}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_4.log {2: 217.4, 1: 217.34, 4: 217.86}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_6.log {2: 217.4, 1: 217.34, 4: 217.86, 6: 217.45}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_5.log {2: 217.4, 1: 217.34, 4: 217.86, 6: 217.45, 5: 217.92}
logs/ngc/tensorflow/bert/bz48/1n2g/bert_b48_fp32_3.log {2: 217.4, 1: 217.34, 4: 217.86, 6: 217.45, 5: 217.92, 3: 217.58}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_2.log {2: 1381.6}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_1.log {2: 1381.6, 1: 1367.84}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_4.log {2: 1381.6, 1: 1367.84, 4: 1397.28}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_6.log {2: 1381.6, 1: 1367.84, 4: 1397.28, 6: 1373.53}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_5.log {2: 1381.6, 1: 1367.84, 4: 1397.28, 6: 1373.53, 5: 1373.08}
logs/ngc/tensorflow/bert/bz48/2n8g/bert_b48_fp32_3.log {2: 1381.6, 1: 1367.84, 4: 1397.28, 6: 1373.53, 5: 1373.08, 3: 1379.36}
{'bert': {'1n1g': {'average_speed': 112.6767,
                   'batch_size_per_device': 48,
                   'median_speed': 112.56,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 217.5917,
                   'batch_size_per_device': 48,
                   'median_speed': 217.51,
                   'speedup': 1.93},
          '1n4g': {'average_speed': 436.7933,
                   'batch_size_per_device': 48,
                   'median_speed': 436.69,
                   'speedup': 3.88},
          '1n8g': {'average_speed': 867.6267,
                   'batch_size_per_device': 48,
                   'median_speed': 867.72,
                   'speedup': 7.71},
          '2n8g': {'average_speed': 1378.7817,
                   'batch_size_per_device': 48,
                   'median_speed': 1376.44,
                   'speedup': 12.23},
          '4n8g': {'average_speed': 2403.965,
                   'batch_size_per_device': 48,
                   'median_speed': 2478.59,
                   'speedup': 22.02}}}
Saving result to ./result/bz48_result.json
```


### 计算规则

#### 1.测速脚本

- extract_tensorflow_logs.py

- extract_tensorflow_logs_time.py

两个脚本略有不同，得到的结果稍有误差：

extract_tensorflow_logs.py根据官方在log中打印的速度，在120个iter中，排除前20iter，取后100个iter的速度做平均；

extract_tensorflow_logs_time.py则根据log中打印出的时间，排除前20iter取后100个iter的实际运行时间计算速度。

README展示的是extract_tensorflow_logs_time.py的计算结果。

#### 2.均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度。

#### 3.加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5

### BERT-Base  FP32

#### batch size=32 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 107.33    | 1       |
| 1        | 4       | 397.71    | 3.71    |
| 1        | 8       | 790.03    | 7.36    |
| 2        | 16      | 1404.04   | 13.08   |
| 4        | 32      | 2727.9    | 25.42   |

#### batch size=48 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 112.76    | 1       |
| 1        | 4       | 430.43    | 3.82    |
| 1        | 8       | 855.45    | 7.59    |
| 2        | 16      | 1576.88   | 13.98   |
| 4        | 32      | 3089.74   | 27.4    |



### BERT-Base  FP16

#### batch size=64 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 183.25    | 1       |
| 1        | 4       | 721.56    | 3.94    |
| 1        | 8       | 1452.59   | 7.93    |
| 2        | 16      | 2653.74   | 14.48   |
| 4        | 32      | 5189.07   | 28.32   |

#### batch size=64 & with xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 422.53    | 1       |
| 1        | 4       | 1552.11   | 3.67    |
| 1        | 8       | 3112.73   | 7.37    |
| 2        | 16      | 5050.86   | 11.95   |
| 4        | 32      | 9409.2    | 22.27   |

#### batch size=96 with xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 468.1     | 1       |
| 1        | 4       | 1766.06   | 3.77    |
| 1        | 8       | 3559.8    | 7.6     |
| 2        | 16      | 5960.14   | 12.73   |
| 4        | 32      | 11650.0   | 24.89   |

注：batch size=96 without xla 情况下会OOM(out of memory)

### 完整日志

- [bert_fp32.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Tensorflow/bert/bert_fp32.zip) 
- [bert_fp16.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Tensorflow/bert/bert_fp16.zip) 
- [bert_fp16_xla.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Tensorflow/bert/bert_fp16_xla.zip) 

