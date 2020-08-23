# NVIDIA-Tensorflow-ResNet50V1.5测评

# Overview
本仓库复现了[NVIDIA官方仓库](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590)中Tensorflow版[ResNet50(v1.5)](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5)的实现，复现的目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力(NVIDIA仓库中，仅给出了单机的测评结果)。

目前，我们仅测试了正常FP32精度下，不加XLA时的情况，后续我们会陆续开展混合精度、XLA等多种方式的测评。



本文主要内容：


-  **Environment** 给出了测评时的硬件系统环境、Docker容器等信息
-  **Quick Start** 介绍了从克隆官方Github仓库到数据集准备的详细过程
-  **Training** 提供了方便易用的测评脚本，覆盖从单机单卡～多机多卡的情形
-  **Result** 提供完整测评log日志，并给出示例代码，用以计算平均速度、加速比，并给出汇总表格



# Environment
## 系统

- 系统：Ubuntu 16.04
- 显卡：Tesla V100（16G）×8
- 显卡驱动：NVIDIA 440.33.01
- CUDA：10.2
- cudnn：7.6.5
## NGC容器

- Ubuntu18.04
- Python 3.6
- **Tensorflow 1.15.2**
- CUDA 10.2.89
- CUDNN 7.6.5
- NCCL 2.6.3
- Horovod 0.19.0
- OpenMPI 3.1.4
- DALI 0.19.0
## Feature support matrix
| Feature | ResNet-50 v1.5 Tensorflow |
| --- | --- |
| Multi-node,multi-gpu training | Yes |
| [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) | Yes |
| Automatic mixed precision (AMP) | No |
| [Horovod](https://github.com/horovod/horovod) | Yes |

# Quick Start
## 项目代码

- [NVIDIA官方仓库](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590)
   - [Resnet50_v1.5项目主页](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5)

下载官方源码：
```shell
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples && checkout fed7ba99cde958fda12c9e81d12b3d7e738e0590
```

1.将本页面scripts文件夹中的脚本：`single_node_train.sh`、`multi_node_train.sh`放入
DeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5/training下；

2.将scripts中的脚本：`SINGLE_NODE_RN50_FP32_1E.sh`、`TWO_NODE_RN50_FP32_1E.sh`和`MULTI_NODE_RN50_FP32_1E.sh`放入resnet50v1.5/training/FP32目录下


## NGC容器
参考[NVIDIA官方Quick start](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide)

**构建项目镜像**


本次测评采用的是NVIDIA官方提供的NGC镜像，您可以在目录DeepLearningExamples/TensorFlow/Classification/ConvNets下运行
```shell
docker build . -t nvidia_rn50_tf:20.03-resnet
```
直接构建本地项目镜像，Dockerfile中将通过`docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3`从网上拉取NVIDIA官方的NGC镜像。

> **本地构建**
>
> 如果您本地有[nvidia:tensorflow](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)的NGC镜像，或者之前通过`docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3`下载过此镜像，则可以修改Dockerfile：
> 这将使得构建项目镜像时，直接从本地已有镜像构建（而不是从网上拉取）。

```shell
# 注释 ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.03-tf1-py3
ARG FROM_IMAGE_NAME=fdc4e72f4c15
```


**启动容器**
```shell
# 构建项目镜像 
# DeepLearningExamples/TensorFlow/Classification/ConvNets目录下
docker build . -t nvidia_rn50_tf:20.03-resnet
# 启动容器
docker  run -it --shm-size=16g --ulimit memlock=-1 --privileged  \
--name tf_resnet  --net host \
--cap-add=IPC_LOCK --device=/dev/infiniband \
-v /datasets/ImageNet/tfrecord:/data/tfrecords \
-d nvidia_rn50_tf:20.03-resnet
```


## 数据集
**tfrecord**

采用imagenet制作的tfrecord格式：train-00000-of-01024,train-00001-of-01024....数据集。参考：nvidia官方的[快速入门指南](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide)

**dali-index**

准备好imagenet数据集后，还需要为dali制作数据集索引：

```shell
# enter docker container
docker exec -it tf_resnet /bin/bash
cd /workspace/rn50v15_tf && mkdir /data/dali_idx
bash ./utils/dali_index.sh /data/tfrecords /data/dali_idx
```
## SSH配置(可选)
单机情况下无需配置ssh服务，需要测试2机、4机等多机情况下，则需要安装docker容器间的ssh服务，配置ssh免密登录，保证分布式horovod/mpi脚本运行时可以在单机上与其他节点互联
**安装ssh服务端**
```shell
docker exec -it tf_resnet  /bin/bash
apt-get update
apt-get install openssh-server
```
**设置免密登录**

- 节点间：/root/.ssh/id_rsa.pub 互相放到/root/.ssh/authorized_keys中；
- 修改sshd中用于docker通信的Port端口号`vim /etc/ssh/sshd_config`
- `service ssh restart`
# Training
集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里设置batch_size=128，从1机1卡～4机32卡进行了一组训练
## 单机
进入容器：
```shell
docker exec -it tf_resnet /bin/bash
cd /workspace/rn50v15
bash ./resnet50v1.5/training/FP32/SINGLE_NODE_RN50_FP32_1E.sh
```
执行脚本，对单机1卡、4卡、8卡分别做6组测试。



## 2机16卡
容器/workspace/rn50v15下执行：`bash resnet50v1.5/training/FP32/TWO_NODE_RN50_FP32_1E.sh`

即可运行2机16卡的训练，同样默认测试6组。

## 4机32卡
容器/workspace/rn50v15下执行：`bash resnet50v1.5/training/FP32/MULTI_NODE_RN50_FP32_1E.sh`

即可运行4机32卡的训练，默认测试6组。


# Result
## 完整日志
[logs.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Tensorflow/resnet50/logs.zip)

## 加速比
执行以下脚本计算各个情况下的加速比：
```shell
python extract_tensorflow_logs.py --log_dir=./logs
```
输出：
```shell
./logs/4n8g/r50_b128_fp32_1.log {1: 9403.78}
./logs/4n8g/r50_b128_fp32_4.log {1: 9403.78, 4: 9477.39}
./logs/4n8g/r50_b128_fp32_2.log {1: 9403.78, 4: 9477.39, 2: 9574.57}
./logs/4n8g/r50_b128_fp32_3.log {1: 9403.78, 4: 9477.39, 2: 9574.57, 3: 9551.9}
./logs/4n8g/r50_b128_fp32_6.log {1: 9403.78, 4: 9477.39, 2: 9574.57, 3: 9551.9, 6: 9631.24}
./logs/4n8g/r50_b128_fp32_5.log {1: 9403.78, 4: 9477.39, 2: 9574.57, 3: 9551.9, 6: 9631.24, 5: 9342.6}
./logs/1n8g/r50_b128_fp32_1.log {1: 2737.81}
./logs/1n8g/r50_b128_fp32_4.log {1: 2737.81, 4: 2696.33}
./logs/1n8g/r50_b128_fp32_2.log {1: 2737.81, 4: 2696.33, 2: 2717.99}
./logs/1n8g/r50_b128_fp32_3.log {1: 2737.81, 4: 2696.33, 2: 2717.99, 3: 2715.18}
./logs/1n8g/r50_b128_fp32_6.log {1: 2737.81, 4: 2696.33, 2: 2717.99, 3: 2715.18, 6: 2725.96}
./logs/1n8g/r50_b128_fp32_5.log {1: 2737.81, 4: 2696.33, 2: 2717.99, 3: 2715.18, 6: 2725.96, 5: 2727.71}
./logs/1n4g/r50_b128_fp32_1.log {1: 1391.53}
./logs/1n4g/r50_b128_fp32_4.log {1: 1391.53, 4: 1393.31}
./logs/1n4g/r50_b128_fp32_2.log {1: 1391.53, 4: 1393.31, 2: 1392.25}
./logs/1n4g/r50_b128_fp32_3.log {1: 1391.53, 4: 1393.31, 2: 1392.25, 3: 1390.17}
./logs/1n4g/r50_b128_fp32_6.log {1: 1391.53, 4: 1393.31, 2: 1392.25, 3: 1390.17, 6: 1391.03}
./logs/1n4g/r50_b128_fp32_5.log {1: 1391.53, 4: 1393.31, 2: 1392.25, 3: 1390.17, 6: 1391.03, 5: 1389.73}
./logs/1n1g/r50_b128_fp32_1.log {1: 362.05}
./logs/1n1g/r50_b128_fp32_4.log {1: 362.05, 4: 362.43}
./logs/1n1g/r50_b128_fp32_2.log {1: 362.05, 4: 362.43, 2: 362.28}
./logs/1n1g/r50_b128_fp32_3.log {1: 362.05, 4: 362.43, 2: 362.28, 3: 362.78}
./logs/1n1g/r50_b128_fp32_6.log {1: 362.05, 4: 362.43, 2: 362.28, 3: 362.78, 6: 362.45}
./logs/1n1g/r50_b128_fp32_5.log {1: 362.05, 4: 362.43, 2: 362.28, 3: 362.78, 6: 362.45, 5: 362.45}
./logs/2n8g/r50_b128_fp32_1.log {1: 5097.79}
./logs/2n8g/r50_b128_fp32_4.log {1: 5097.79, 4: 5018.55}
./logs/2n8g/r50_b128_fp32_2.log {1: 5097.79, 4: 5018.55, 2: 5063.02}
./logs/2n8g/r50_b128_fp32_3.log {1: 5097.79, 4: 5018.55, 2: 5063.02, 3: 5107.27}
./logs/2n8g/r50_b128_fp32_6.log {1: 5097.79, 4: 5018.55, 2: 5063.02, 3: 5107.27, 6: 5125.81}
./logs/2n8g/r50_b128_fp32_5.log {1: 5097.79, 4: 5018.55, 2: 5063.02, 3: 5107.27, 6: 5125.81, 5: 5101.06}
{'r50': {'1n1g': {'average_speed': 362.41,
                  'batch_size_per_device': 128,
                  'speedup': 1.0},
         '1n4g': {'average_speed': 1391.34,
                  'batch_size_per_device': 128,
                  'speedup': 3.84},
         '1n8g': {'average_speed': 2720.16,
                  'batch_size_per_device': 128,
                  'speedup': 7.5},
         '2n8g': {'average_speed': 5085.58,
                  'batch_size_per_device': 128,
                  'speedup': 14.03},
         '4n8g': {'average_speed': 9496.91,
                  'batch_size_per_device': 128,
                  'speedup': 26.2}}}
Saving result to ./result/logs_result.json
```
## ResNet50 V1.5 bsz = 128

### FP32 & Without XLA

| 节点数 | GPU数 | samples/s(OneFlow) | 加速比 | samples/s(tensorflow) | 加速比 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 383.76 | 1 | 362.41 | 1 |
| 1 | 4 | 1497.62 | 3.90 | 1391.34 | 3.84 |
| 1 | 8 | 2942.32 | 7.67 | 2720.16 | 7.5 |
| 2 | 16 | 5839.05 | 15.22 | 5085.58 | 14.03 |
| 4 | 32 | 11548.45 | 30.09 | 9496.91 | 26.2 |



附：[NVIDIA DGX-1 (8x V100 16G)官方测试结果](https://github.com/NVIDIA/DeepLearningExamples/tree/709456cdd7a0f2ae03fe42846ec6a24dceee536e/TensorFlow/Classification/RN50v1.5#nvidia-dgx-1-8x-v100-16g-1)

| **FP32 img/s** | **number of GPUs** |
| --- | --- |
| 364.9 | **1** |
| 1419.4 | **4** |
| 2778.5 | **8** |



