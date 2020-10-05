# NVIDIA-Tensorflow-ResNet50V1.5测评

# Overview

本仓库复现了[NVIDIA官方仓库](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590)中TensorFlow版[ResNet50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5)，目的在于速度测评，得到1机、2机、4机情况下的吞吐率及加速比，评判框架在分布式训练情况下的横向拓展能力。

目前，测试覆盖了 FP32精度、FP16混合精度以及XLA，后续将持续维护增加更多方式的测评。



# Environment

## 系统

- 系统：Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- 显卡：Tesla V100-SXM2-16GB x 8
- 显卡驱动：NVIDIA 440.33.01
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

| Feature                                                      | ResNet-50 v1.5 TensorFlow |
| ------------------------------------------------------------ | ------------------------- |
| [Horovod Multi-gpu](https://github.com/horovod/horovod)      | Yes                       |
| [Horovod Multi-node](https://github.com/horovod/horovod)     | Yes                       |
| Automatic mixed precision (AMP)                              | Yes                       |
| [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) | Yes                       |

# Quick Start

## 项目代码

- [NVIDIA官方仓库](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590)
  - [Resnet50_v1.5项目主页](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5)

下载官方源码：

```shell
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples && git checkout fed7ba99cde958fda12c9e81d12b3d7e738e0590
```

将本页面scripts文件夹下的脚本放入：
 `/DeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5/training`目录下。




## NGC容器

参考[NVIDIA官方Quick start](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide)

**构建项目镜像**


本次测评采用的是NVIDIA官方提供的NGC 20.03镜像，您可以在目录`/DeepLearningExamples/TensorFlow/Classification/ConvNets`下运行

```shell
docker build . -t nvidia_rn50_tf:20.03-resnet
```

直接构建本地项目镜像，Dockerfile中将通过`docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3`从网上拉取NVIDIA官方的NGC镜像。

> **本地构建**
>
> 如果您之前通过`docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3`下载过此镜像，或者本地有[nvidia:tensorflow](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)的NGC镜像，则可以修改Dockerfile：
>
> ```shell
> # 注释 ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.03-tf1-py3
> ARG FROM_IMAGE_NAME=fdc4e72f4c15
> ```
>
> 这将使得构建项目镜像时，直接从本地已有镜像构建（而不是从网上拉取）。

**启动容器**

```shell
# 构建项目镜像 
# /DeepLearningExamples/TensorFlow/Classification/ConvNets目录下
docker build . -t nvidia_rn50_tf:20.03-resnet
# 启动容器
docker  run -it --shm-size=16g --ulimit memlock=-1 --privileged  \
--name tf_resnet  --net host \
--cap-add=IPC_LOCK --device=/dev/infiniband \
-v /datasets/ImageNet/tfrecord:/data/tfrecords \
-d nvidia_rn50_tf:20.03-resnet
```


## 数据集

**TFRecord**

采用ImageNet制作的`tfrecord`格式：train-00000-of-01024,train-00001-of-01024....数据集。参考：NVIDIA官方的[快速入门指南](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide)

**dali-index**

准备好ImageNet数据集后，还需要为DALI制作数据集索引：

```shell
# enter docker container
docker exec -it tf_resnet /bin/bash
cd /workspace/rn50v15_tf && mkdir /data/dali_idx
bash ./utils/dali_index.sh /data/tfrecords /data/dali_idx
```

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

每个节点有8张显卡，这里设置batch_size=128，从1机1卡～4机32卡进行了6次训练。

## 单机

进入容器：

```shell
docker exec -it tf_resnet /bin/bash
cd /workspace/rn50v15
bash resnet50v1.5/training/run_single_node.sh
```

执行脚本，即可运行单机1卡、4卡、8卡的训练，分别测试5次。默认测试FP32、batch size=128的情况。

### 混合精度

可以通过参数指定进行FP16混合精度的训练，如以下命令将进行bath size=224的FP16混合精度训练：

`bash resnet50v1.5/training/run_single_node.sh 224 amp`



## 2机16卡

容器/workspace/rn50v15下执行：`bash resnet50v1.5/training/run_two_node.sh`

即可运行2机16卡的训练，同样默认测试5次。

### 混合精度

可以通过参数指定进行FP16混合精度的训练，如以下命令将进行bath size=224的2机FP16混合精度训练：

`bash resnet50v1.5/training/run_two_node.sh 224  amp`



## 4机32卡

容器/workspace/rn50v15下执行：`bash resnet50v1.5/training/run_multi_node.sh`

即可运行4机32卡的训练，默认测试5次。

## 混合精度&XLA

### FP6混合精度

可以通过参数指定进行FP16混合精度的训练，如以下命令将进行bath size=224的FP6混合精度训练

- 单机混合精度训练：

  - `bash resnet50v1.5/training/run_single_node.sh 224 amp`

- 2机混合精度训练：

  - `bash resnet50v1.5/training/run_two_node.sh 224  amp`

- 4机混合精度训练：

  - `bash resnet50v1.5/training/run_multi_node.sh 224  amp`

  

### 启用XLA

所有训练默认使用dali，所以以上脚本中都加有参数USE_DALI=1：

```shel
USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE}   ${DATA_DIR}    1   $NUM_STEP   $BATCH_SIZE   $DTYPE   $i
```

同样，需要启用XLA只需要加上USE_XLA=1即可：

```she
USE_DALI=1 USE_XLA=1 bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ....
```




# Results

## 吞吐率及加速比

执行以下命令，即可计算各种测试配置下的吞吐率及加速比：

```shell
python extract_tensorflow_logs.py --log_dir=logs/ngc/tensorflow/resnet50 --batch_size_per_device=128
```

输出：

```shell
logs/ngc/tensorflow/resnet50/4n8g/r50_b128_fp32_1.log {1: 9403.78}
logs/ngc/tensorflow/resnet50/4n8g/r50_b128_fp32_4.log {1: 9403.78, 4: 9477.39}
logs/ngc/tensorflow/resnet50/4n8g/r50_b128_fp32_2.log {1: 9403.78, 4: 9477.39, 2: 9574.57}
logs/ngc/tensorflow/resnet50/4n8g/r50_b128_fp32_3.log {1: 9403.78, 4: 9477.39, 2: 9574.57, 3: 9551.9}
logs/ngc/tensorflow/resnet50/4n8g/r50_b128_fp32_6.log {1: 9403.78, 4: 9477.39, 2: 9574.57, 3: 9551.9, 6: 9631.24}
logs/ngc/tensorflow/resnet50/4n8g/r50_b128_fp32_5.log {1: 9403.78, 4: 9477.39, 2: 9574.57, 3: 9551.9, 6: 9631.24, 5: 9342.6}
logs/ngc/tensorflow/resnet50/1n8g/r50_b128_fp32_1.log {1: 2737.81}
logs/ngc/tensorflow/resnet50/1n8g/r50_b128_fp32_4.log {1: 2737.81, 4: 2696.33}
logs/ngc/tensorflow/resnet50/1n8g/r50_b128_fp32_2.log {1: 2737.81, 4: 2696.33, 2: 2718.0}
logs/ngc/tensorflow/resnet50/1n8g/r50_b128_fp32_3.log {1: 2737.81, 4: 2696.33, 2: 2718.0, 3: 2715.18}
logs/ngc/tensorflow/resnet50/1n8g/r50_b128_fp32_6.log {1: 2737.81, 4: 2696.33, 2: 2718.0, 3: 2715.18, 6: 2725.96}
logs/ngc/tensorflow/resnet50/1n8g/r50_b128_fp32_5.log {1: 2737.81, 4: 2696.33, 2: 2718.0, 3: 2715.18, 6: 2725.96, 5: 2727.71}
logs/ngc/tensorflow/resnet50/1n4g/r50_b128_fp32_1.log {1: 1391.53}
logs/ngc/tensorflow/resnet50/1n4g/r50_b128_fp32_4.log {1: 1391.53, 4: 1393.31}
logs/ngc/tensorflow/resnet50/1n4g/r50_b128_fp32_2.log {1: 1391.53, 4: 1393.31, 2: 1392.25}
logs/ngc/tensorflow/resnet50/1n4g/r50_b128_fp32_3.log {1: 1391.53, 4: 1393.31, 2: 1392.25, 3: 1390.17}
logs/ngc/tensorflow/resnet50/1n4g/r50_b128_fp32_6.log {1: 1391.53, 4: 1393.31, 2: 1392.25, 3: 1390.17, 6: 1391.03}
logs/ngc/tensorflow/resnet50/1n4g/r50_b128_fp32_5.log {1: 1391.53, 4: 1393.31, 2: 1392.25, 3: 1390.17, 6: 1391.03, 5: 1389.73}
logs/ngc/tensorflow/resnet50/1n1g/r50_b128_fp32_1.log {1: 362.05}
logs/ngc/tensorflow/resnet50/1n1g/r50_b128_fp32_4.log {1: 362.05, 4: 362.43}
logs/ngc/tensorflow/resnet50/1n1g/r50_b128_fp32_2.log {1: 362.05, 4: 362.43, 2: 362.28}
logs/ngc/tensorflow/resnet50/1n1g/r50_b128_fp32_3.log {1: 362.05, 4: 362.43, 2: 362.28, 3: 362.78}
logs/ngc/tensorflow/resnet50/1n1g/r50_b128_fp32_6.log {1: 362.05, 4: 362.43, 2: 362.28, 3: 362.78, 6: 362.45}
logs/ngc/tensorflow/resnet50/1n1g/r50_b128_fp32_5.log {1: 362.05, 4: 362.43, 2: 362.28, 3: 362.78, 6: 362.45, 5: 362.45}
logs/ngc/tensorflow/resnet50/2n8g/r50_b128_fp32_1.log {1: 5097.79}
logs/ngc/tensorflow/resnet50/2n8g/r50_b128_fp32_4.log {1: 5097.79, 4: 5018.54}
logs/ngc/tensorflow/resnet50/2n8g/r50_b128_fp32_2.log {1: 5097.79, 4: 5018.54, 2: 5063.02}
logs/ngc/tensorflow/resnet50/2n8g/r50_b128_fp32_3.log {1: 5097.79, 4: 5018.54, 2: 5063.02, 3: 5107.27}
logs/ngc/tensorflow/resnet50/2n8g/r50_b128_fp32_6.log {1: 5097.79, 4: 5018.54, 2: 5063.02, 3: 5107.27, 6: 5125.81}
logs/ngc/tensorflow/resnet50/2n8g/r50_b128_fp32_5.log {1: 5097.79, 4: 5018.54, 2: 5063.02, 3: 5107.27, 6: 5125.81, 5: 5101.06}
{'r50': {'1n1g': {'average_speed': 362.41,
                  'batch_size_per_device': 128,
                  'median_speed': 362.44,
                  'speedup': 1.0},
         '1n4g': {'average_speed': 1391.34,
                  'batch_size_per_device': 128,
                  'median_speed': 1391.28,
                  'speedup': 3.84},
         '1n8g': {'average_speed': 2720.16,
                  'batch_size_per_device': 128,
                  'median_speed': 2721.98,
                  'speedup': 7.51},
         '2n8g': {'average_speed': 5085.58,
                  'batch_size_per_device': 128,
                  'median_speed': 5099.42,
                  'speedup': 14.07},
         '4n8g': {'average_speed': 9496.91,
                  'batch_size_per_device': 128,
                  'median_speed': 9514.64,
                  'speedup': 26.25}}}
Saving result to ./result/resnet50_result.json
```

### 计算规则

#### 1.测速脚本

- extract_tensorflow_logs.py

- extract_tensorflow_logs_time.py

两个脚本略有不同，得到的结果稍有误差：

extract_tensorflow_logs.py根据官方在log中打印的速度，在120个iter中，排除前20iter，取后100个iter的速度做平均；

extract_tensorflow_logs_time.py则根据log中打印出的时间，排除前20iter取后100个iter的实际运行时间计算速度。

README展示的是extract_tensorflow_logs.py的计算结果。

#### 2.均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度。

#### 3.加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5



## ResNet50 V1.5 bsz = 128

### FP32 & Without XLA

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 362.44    | 1.00    |
| 1        | 4       | 1391.28   | 3.84    |
| 1        | 8       | 2721.98   | 7.51    |
| 2        | 16      | 5099.42   | 14.07   |
| 4        | 32      | 9514.64   | 26.25   |



## ResNet50 V1.5 bsz = 224

### FP16 & Without XLA

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 945.18    | 1       |
| 1        | 4       | 3546.02   | 3.75    |
| 1        | 8       | 6903.42   | 7.3     |
| 2        | 16      | 12021.09  | 12.72   |
| 4        | 32      | 24734.22  | 26.17   |

### FP16 & With XLA

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 1198.55   | 1       |
| 1        | 4       | 4360.83   | 3.64    |
| 1        | 8       | 8588.45   | 7.17    |
| 2        | 16      | 14931.03  | 12.46   |
| 4        | 32      | 29171.69  | 24.34   |

[NVIDIA DGX-1 (8x V100 16G)官方测试结果](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5#training-performance-nvidia-dgx-1-8x-v100-16g)



注意：

1.官方测速[采用的脚本](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5#benchmarking) 是纯为了跑速度的，很多参数并没有和训练时的参数对齐（label_smoothing设为0、use_cosine_lr=False、use_static_loss_scaling=False等）而[官方amp的训练脚本](https://github.com/NVIDIA/DeepLearningExamples/blob/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5/training/GENERIC.sh)中这些参数都是有的。我们测速的原则是真实反应各框架，在真实训练过程中的速度，所以加上了这些参数。

2.本次测速时最大能跑到的batch size为224，跑官方宣称的256时会OOM(out of memory)，故理论上batch size=224的数据相比256会差一些；

3.速度差异的原因还有可能是机器环境不同，数据集制作方式不同，后期考虑用更统一和规范的数据集进行测试。





## 完整日志

-  [resnet50_fp32.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Tensorflow/resnet50/resnet50_fp32.zip) 
-  [resnet50_fp16.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Tensorflow/resnet50/resnet50_fp16.zip) 
-  [resnet50_fp16_xla.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Tensorflow/resnet50/resnet50_fp16_xla.zip)

