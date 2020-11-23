# NVIDIA/DeepLearningExamples PyTorch ResNet50 v1.5 测评


## 概述 Overview


本测试基于 [NVIDIA/DeepLearningExamples/Classification/ConvNets/](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/Classification/ConvNets) 仓库中提供的 PyTorch 框架的 [ResNet50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/Classification/ConvNets/resnet50v1.5) 实现，在 NVIDIA 官方提供的 [20.03 NGC 镜像及其衍生容器](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags)中进行单机单卡、单机多卡、多机多卡的结果复现及速度评测，评判框架在分布式训练情况下的横向拓展能力。


目前，该测试覆盖 FP32 及混合精度，后续将持续维护，增加使用其他优化方式的测评。


## 内容目录 Table Of Content


- [NVIDIA/DeepLearningExamples PyTorch ResNet50 v1.5 测评](#nvidia-deeplearningexamples-pytorch-resnet50-v15---)
  * [概述 Overview](#---overview)
  * [内容目录 Table Of Content](#-----table-of-content)
  * [环境 Environment](#---environment)
    + [系统](#--)
      - [硬件](#--)
      - [软件](#--)
    + [NGC 容器](#ngc---)
      - [Feature support matrix](#feature-support-matrix)
  * [快速开始 Quick Start](#-----quick-start)
    + [1. 前期准备](#1-----)
      - [数据集](#---)
      - [镜像及容器](#-----)
      - [安装 IB 驱动](#---ib---)
    + [2. 运行测试](#2-----)
      - [单机测试](#----)
      - [多机测试](#----)
        * [两机测试](#----)
        * [多机测试](#-----1)
    + [3. 数据处理](#3-----)
  * [性能结果 Performance](#-----performance)
    + [FP32](#fp32)
    + [ResNet50 v1.5 batch_size = 128](#resnet50-v15-batch-size---128)
    + [AMP](#amp)
    + [ResNet50 v1.5 batch_size = 256](#resnet50-v15-batch-size---256)


## 环境 Environment


### 系统


- #### 硬件


  - GPU：Tesla V100-SXM2-16GB x 8


- #### 软件


  - 驱动：NVIDIA 440.33.01


  - 系统：[ Ubuntu 16.04](http://releases.ubuntu.com/16.04/)

  - CUDA：10.2


  - cuDNN：7.6.5


### NGC 容器


- 系统：[ Ubuntu 18.04](http://releases.ubuntu.com/18.04/)

- CUDA 10.2.89


- cuDNN 7.6.5


- NCCL：2.5.6


- PyTorch：1.5.0a0+8f84ded


- OpenMPI 3.1.4


- DALI 0.19.0


- Python：3.6.9


  更多容器细节请参考 [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html)。


  #### Feature support matrix

  | Feature                                                      | ResNet50 v1.5 PyTorch |
  | ------------------------------------------------------------ | --------------------- |
  | Multi-gpu training                                           | Yes                   |
  | Multi-node training                                          | Yes                   |
  | [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) | Yes                   |
  | Automatic mixed precision (AMP)                              | Yes                   |



## 快速开始 Quick Start


### 1. 前期准备


- #### 数据集


根据 [Convolutional Networks for Image Classification in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/Classification/ConvNets) 准备 ImageNet 数据集，只需下载、解压 train、validation 数据集到对应路径即可，使用原始图片进行训练。


- #### 镜像及容器


拉取 NGC 20.03 的镜像、搭建容器，进入容器环境。


```
# 下载镜像
docker pull nvcr.io/nvidia/pytorch:20.03-py3 

# 启动容器
docker run -it --shm-size=16g --ulimit memlock=-1 --privileged  \
--name pt_bert  --net host \
-v ./data:/data/ \
-d pytorch:20.03-py3 
```

- #### 安装 IB 驱动


测试机器上的容器环境内未查找到 IB 驱动，会导致测试时 NCCL 库只能使用 Socket 通信，无法达到最佳测试效果，因此需要额外安装，首先安装依赖


```
apt install dpatch libelf1 libmnl0 libltdl-dev lsof chrpath debhelper pciutils tk bison graphviz ethtool kmod gfortran swig flex tcl
```

更换为阿里云源


```
cp /etc/apt/sources.list /etc/apt/sources.list.bak

vim /etc/apt/sources.list
```

将下列源址复制进 /etc/apt/sources.list 中


```
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse

 

deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse

 

deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
```

更新源


```
apt-get update
```

下载[软件 MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64.tar 源码包](http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64.tar)并解压


```
wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64.tar && tar -xvf MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64.tar
```

进入源码包路径，安装


```
cd MLNX_OFED_LINUX-4.9-0.1.7.0-ubuntu18.04-x86_64 && ./mlnxofedinstall --user-space-only --without-fw-update --all --force 
```

安装时出现


```
......
Installing srptools-41mlnx1...
Installing mlnx-ethtool-5.4...
Installing mlnx-iproute2-5.4.0...
Installing neohost-backend-1.5.0...
Failed to install neohost-backend DEB
Collecting debug info...
See /tmp/MLNX_OFED_LINUX.24525.logs/neohost-backend.debinstall.log
```

可以忽略。完成后，检查驱动是否安装成功


```
ibstat
```

打印


```
root@VS002:/workspace/rn50# ibstat
CA 'mlx5_0'
        CA type: MT4115
        Number of ports: 1
        Firmware version: 12.27.1016
        Hardware version: 0
        Node GUID: 0x506b4b0300f37674
        System image GUID: 0x506b4b0300f37674
        Port 1:
                State: Active
                Physical state: LinkUp
                Rate: 100
                Base lid: 56
                LMC: 0
                SM lid: 27
                Capability mask: 0x2651e848
                Port GUID: 0x506b4b0300f37674
                Link layer: InfiniBand
```

即为成功。




### 2. 运行测试


本次测试集群中有 4 台节点：


- NODE1=10.11.0.2

- NODE2=10.11.0.3

- NODE3=10.11.0.4

- NODE4=10.11.0.5


每个节点有 8 张 V100 显卡， 每张显卡显存 16 G。


- #### 单机测试


在节点 1 的容器内下载本仓库源码：


````
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 /DLPerf/NVIDIADeepLearningExamples/PyTorch/resnet50v1.5/scripts 路径源码放至 /workspace/rn50 下，执行脚本


```
bash scripts/run_single_node.sh
```

针对单机单卡、2 卡、4 卡、8 卡， batch_size 取 128 等情况进行测试，并将 log 信息保存在当前目录的 /ngc/pytorch/ 对应分布式配置路径中。


- #### 多机测试


多机测试，一定要确保数据集存在各节点测试机器的相同路径下，各脚本的行为要一致，尤其是修改要保持同步。


典型地，多机测试时，需要在 /workspace/rn50/main.py 中 310 行的 `torch.cuda.set_device(args.gpu)` 下方增加 `args.gpu = torch.cuda.device_count()`，即


```
308     if args.distributed:
309         args.gpu = args.local_rank % torch.cuda.device_count()
310         torch.cuda.set_device(args.gpu)
311         args.gpu = torch.cuda.device_count() # modify here
312         dist.init_process_group(backend="nccl", init_method="env://")
313         args.world_size = torch.distributed.get_world_size()
```

4 台机器都需要增加。


另外，需要测试混合精度（AMP）时，应该修改 `PREC` 的精度选项（`amp`）。


- ##### 两机测试


以 NODE1 和 NODE2 为例，run_two_nodes.sh 脚本已填入 2 台机器对应的 IP 及端口号，NODE1 上的脚本 single_node_train.sh 中 `--node_rank` 默认为 0，还需自行将 NODE2 机器上相同路径下的脚本 37 行 `--node_rank` 改为 1，在 2 台机器上同时运行脚本，打印 log 如下：


```
+ '[' -z ngc/pytorch/2n8g/r50_b128_fp32_5.log ']'
+ tee ngc/pytorch/2n8g/r50_b128_fp32_5.log
+ python /workspace/rn50/multiproc.py --nnodes 2 --node_rank 0 --nproc_per_node 8 --master_addr 10.11.0.2 --master_port=22333 /workspace/rn50/main.py --data-backend dali-cpu --raport-file /workspace/rn50/raport.json -j8 -p 1 --lr 1.024 --optimizer-batch-size -1 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.125 --wd 3.0517578125e-05 --workspace /workspace/rn50 -b 128 --epochs 1 --prof 121 --training-only --no-checkpoints /data/image
=> creating model '('resnet50', 'fanin', 1000)'
Version: {'net': <class 'image_classification.resnet.ResNet'>, 'block': <class 'image_classification.resnet.Bottleneck'>, 'layers': [3, 4, 6, 3], 'widths': [64, 128, 256, 512], 'expansion': 4}
Config: {'conv': <class 'torch.nn.modules.conv.Conv2d'>, 'conv_init': 'fan_in', 'nonlinearity': 'relu', 'last_bn_0_init': False, 'activation': <function <lambda> at 0x7f9ab49a19d8>}
read 886372 files from 698 directories
read 50000 files from 1000 directories
DLL 2020-09-15 14:28:53.498443 - PARAMETER data : /data/image  data_backend : dali-cpu  arch : resnet50  model_config : fanin  num_classes : 1000  workers : 8  epochs : 1  run_epochs : -1  batch_size : 128  optimizer_batch_size : -1  lr : 1.024  lr_schedule : cosine  warmup : 8  label_smoothing : 0.1  mixup : 0.0  momentum : 0.125  weight_decay : 3.0517578125e-05  bn_weight_decay : False  nesterov : False  print_freq : 1  resume : None  pretrained_weights :   fp16 : False  static_loss_scale : 1  dynamic_loss_scale : False  prof : 121  amp : False  seed : None  gather_checkpoints : False  raport_file : /workspace/rn50/raport.json  evaluate : False  training_only : True  save_checkpoints : False  checkpoint_filename : checkpoint.pth.tar  workspace : /workspace/rn50  memory_format : nchw  distributed : True  local_rank : 0  gpu : 8  world_size : 16 
 ! Weight decay NOT applied to BN parameters 
98
63
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
RUNNING EPOCHS FROM 0 TO 1
DLL 2020-09-15 14:29:13.473765 - Epoch: 0 Iteration: 1  train.loss : 7.09304  train.total_ips : 267.44 img/s
DLL 2020-09-15 14:29:14.427490 - Epoch: 0 Iteration: 2  train.loss : 6.93504  train.total_ips : 4294.91 img/s
DLL 2020-09-15 14:29:15.018767 - Epoch: 0 Iteration: 3  train.loss : 6.85957  train.total_ips : 6928.79 img/s
DLL 2020-09-15 14:29:15.387051 - Epoch: 0 Iteration: 4  train.loss : 6.80420  train.total_ips : 11124.52 img/s
DLL 2020-09-15 14:29:15.791445 - Epoch: 0 Iteration: 5  train.loss : 6.75717  train.total_ips : 10131.38 img/s
DLL 2020-09-15 14:29:16.161849 - Epoch: 0 Iteration: 6  train.loss : 6.73528  train.total_ips : 11064.93 img/s
DLL 2020-09-15 14:29:16.538463 - Epoch: 0 Iteration: 7  train.loss : 6.72415  train.total_ips : 10879.48 img/s
DLL 2020-09-15 14:29:16.946395 - Epoch: 0 Iteration: 8  train.loss : 6.71593  train.total_ips : 10044.52 img/s
DLL 2020-09-15 14:29:17.334849 - Epoch: 0 Iteration: 9  train.loss : 6.69474  train.total_ips : 10547.30 img/s
DLL 2020-09-15 14:29:17.744287 - Epoch: 0 Iteration: 10  train.loss : 6.69965  train.total_ips : 10007.33 img/s
DLL 2020-09-15 14:29:18.159381 - Epoch: 0 Iteration: 11  train.loss : 6.69236  train.total_ips : 9872.60 img/s
DLL 2020-09-15 14:29:18.553769 - Epoch: 0 Iteration: 12  train.loss : 6.67351  train.total_ips : 10393.54 img/s
```



- ##### 多机测试


以本集群为例，最多支持 4 机 32 卡，run_multi_nodes.sh 脚本已设置 NODE1 为 master node，设置好其 IP 及端口号，还需自行将 NODE3 机器上相同路径下的脚本 37 行 `--node_rank` 中的改为 2， NODE4 的 `--node_rank` 改为 3，在 4 台机器上同时运行脚本，打印 log 如下：


```
+ '[' -z ngc/pytorch/4n8g/r50_b128_fp32_5.log ']'
+ tee ngc/pytorch/4n8g/r50_b128_fp32_5.log
+ python /workspace/rn50/multiproc.py --nnodes 4 --node_rank 0 --nproc_per_node 8 --master_addr 10.11.0.2 --master_port=22333 /workspace/rn50/main.py --data-backend dali-cpu --raport-file /workspace/rn50/raport.json -j8 -p 1 --lr 1.024 --optimizer-batch-size -1 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.125 --wd 3.0517578125e-05 --workspace /workspace/rn50 -b 128 --epochs 1 --prof 121 --training-only --no-checkpoints /data/image
=> creating model '('resnet50', 'fanin', 1000)'
Version: {'net': <class 'image_classification.resnet.ResNet'>, 'block': <class 'image_classification.resnet.Bottleneck'>, 'layers': [3, 4, 6, 3], 'widths': [64, 128, 256, 512], 'expansion': 4}
Config: {'conv': <class 'torch.nn.modules.conv.Conv2d'>, 'conv_init': 'fan_in', 'nonlinearity': 'relu', 'last_bn_0_init': False, 'activation': <function <lambda> at 0x7f9ab49a19d8>}
read 886372 files from 698 directories
read 50000 files from 1000 directories
DLL 2020-09-15 14:28:53.498443 - PARAMETER data : /data/image  data_backend : dali-cpu  arch : resnet50  model_config : fanin  num_classes : 1000  workers : 8  epochs : 1  run_epochs : -1  batch_size : 128  optimizer_batch_size : -1  lr : 1.024  lr_schedule : cosine  warmup : 8  label_smoothing : 0.1  mixup : 0.0  momentum : 0.125  weight_decay : 3.0517578125e-05  bn_weight_decay : False  nesterov : False  print_freq : 1  resume : None  pretrained_weights :   fp16 : False  static_loss_scale : 1  dynamic_loss_scale : False  prof : 121  amp : False  seed : None  gather_checkpoints : False  raport_file : /workspace/rn50/raport.json  evaluate : False  training_only : True  save_checkpoints : False  checkpoint_filename : checkpoint.pth.tar  workspace : /workspace/rn50  memory_format : nchw  distributed : True  local_rank : 0  gpu : 8  world_size : 32 
 ! Weight decay NOT applied to BN parameters 
98
63
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
RUNNING EPOCHS FROM 0 TO 1
DLL 2020-09-15 14:29:13.473765 - Epoch: 0 Iteration: 1  train.loss : 7.09304  train.total_ips : 267.44 img/s
DLL 2020-09-15 14:29:14.427490 - Epoch: 0 Iteration: 2  train.loss : 6.93504  train.total_ips : 4294.91 img/s
DLL 2020-09-15 14:29:15.018767 - Epoch: 0 Iteration: 3  train.loss : 6.85957  train.total_ips : 6928.79 img/s
DLL 2020-09-15 14:29:15.387051 - Epoch: 0 Iteration: 4  train.loss : 6.80420  train.total_ips : 11124.52 img/s
DLL 2020-09-15 14:29:15.791445 - Epoch: 0 Iteration: 5  train.loss : 6.75717  train.total_ips : 10131.38 img/s
DLL 2020-09-15 14:29:16.161849 - Epoch: 0 Iteration: 6  train.loss : 6.73528  train.total_ips : 11064.93 img/s
DLL 2020-09-15 14:29:16.538463 - Epoch: 0 Iteration: 7  train.loss : 6.72415  train.total_ips : 10879.48 img/s
DLL 2020-09-15 14:29:16.946395 - Epoch: 0 Iteration: 8  train.loss : 6.71593  train.total_ips : 10044.52 img/s
DLL 2020-09-15 14:29:17.334849 - Epoch: 0 Iteration: 9  train.loss : 6.69474  train.total_ips : 10547.30 img/s
DLL 2020-09-15 14:29:17.744287 - Epoch: 0 Iteration: 10  train.loss : 6.69965  train.total_ips : 10007.33 img/s
DLL 2020-09-15 14:29:18.159381 - Epoch: 0 Iteration: 11  train.loss : 6.69236  train.total_ips : 9872.60 img/s
DLL 2020-09-15 14:29:18.553769 - Epoch: 0 Iteration: 12  train.loss : 6.67351  train.total_ips : 10393.54 img/s
```



### 3. 数据处理


测试进行了多组训练（本测试中取 5 次），每次训练过程取第 1 个 epoch 的前 150 iter，计算训练速度时取后 100 iter 的数据，以降低抖动。最后将 5 次训练的结果取中位数得到最终速度，并最终以此数据计算加速比。


运行 /DLPerf/NVIDIADeepLearningExamples/PyTorch/BERT/extract_pytorch_logs_time.py，即可得到针对不同配置测试结果 log 数据处理的结果： 


```
python extract_pytorch_logs_time.py --log_dir /workspace/rn50/scripts/j5_amp_ngc/pytorch/ --warmup_batches 20 --train_batches 120 --batch_size_per_device 256
```

结果打印如下


```
/workspace/rn50/scripts/j5_amp_ngc/pytorch/4n8g/r50_b256_amp_3.log {3: 22978.13}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/4n8g/r50_b256_amp_5.log {3: 22978.13, 5: 22053.98}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/4n8g/r50_b256_amp_2.log {3: 22978.13, 5: 22053.98, 2: 22551.16}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/4n8g/r50_b256_amp_4.log {3: 22978.13, 5: 22053.98, 2: 22551.16, 4: 23049.75}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/4n8g/r50_b256_amp_1.log {3: 22978.13, 5: 22053.98, 2: 22551.16, 4: 23049.75, 1: 22364.13}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n1g/r50_b256_fp16_4.log {4: 802.4}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n1g/r50_b256_fp16_1.log {4: 802.4, 1: 803.69}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n1g/r50_b256_fp16_5.log {4: 802.4, 1: 803.69, 5: 802.9}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n1g/r50_b256_fp16_3.log {4: 802.4, 1: 803.69, 5: 802.9, 3: 803.66}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n1g/r50_b256_fp16_2.log {4: 802.4, 1: 803.69, 5: 802.9, 3: 803.66, 2: 793.9}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/2n8g/r50_b256_amp_3.log {3: 11991.94}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/2n8g/r50_b256_amp_5.log {3: 11991.94, 5: 11964.81}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/2n8g/r50_b256_amp_2.log {3: 11991.94, 5: 11964.81, 2: 11896.95}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/2n8g/r50_b256_amp_4.log {3: 11991.94, 5: 11964.81, 2: 11896.95, 4: 11999.35}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/2n8g/r50_b256_amp_1.log {3: 11991.94, 5: 11964.81, 2: 11896.95, 4: 11999.35, 1: 12046.71}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n8g/r50_b256_fp16_4.log {4: 6173.05}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n8g/r50_b256_fp16_1.log {4: 6173.05, 1: 6135.08}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n8g/r50_b256_fp16_5.log {4: 6173.05, 1: 6135.08, 5: 6160.56}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n8g/r50_b256_fp16_3.log {4: 6173.05, 1: 6135.08, 5: 6160.56, 3: 6154.66}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n8g/r50_b256_fp16_2.log {4: 6173.05, 1: 6135.08, 5: 6160.56, 3: 6154.66, 2: 6130.69}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n4g/r50_b256_fp16_4.log {4: 3137.67}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n4g/r50_b256_fp16_1.log {4: 3137.67, 1: 3129.63}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n4g/r50_b256_fp16_5.log {4: 3137.67, 1: 3129.63, 5: 3144.55}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n4g/r50_b256_fp16_3.log {4: 3137.67, 1: 3129.63, 5: 3144.55, 3: 3137.44}
/workspace/rn50/scripts/j5_amp_ngc/pytorch/1n4g/r50_b256_fp16_2.log {4: 3137.67, 1: 3129.63, 5: 3144.55, 3: 3137.44, 2: 3119.23}
{'r50': {'1n1g': {'average_speed': 801.31,
                  'batch_size_per_device': 256,
                  'median_speed': 802.9,
                  'speedup': 1.0},
         '1n4g': {'average_speed': 3133.7,
                  'batch_size_per_device': 256,
                  'median_speed': 3137.44,
                  'speedup': 3.91},
         '1n8g': {'average_speed': 6150.81,
                  'batch_size_per_device': 256,
                  'median_speed': 6154.66,
                  'speedup': 7.67},
         '2n8g': {'average_speed': 11979.95,
                  'batch_size_per_device': 256,
                  'median_speed': 11991.94,
                  'speedup': 14.94},
         '4n8g': {'average_speed': 22599.43,
                  'batch_size_per_device': 256,
                  'median_speed': 22551.16,
                  'speedup': 28.09}}}
Saving result to ./result/_result.json
```



## 性能结果 Performance


该小节提供针对 NVIDIA PyTorch 框架的 ResNet50 v1.5 模型使用 IB（Infinite Band）网络单多机测试的性能结果和完整 log 日志。


### FP32 


- ### ResNet50 v1.5 batch_size = 128


| node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |

| -------- | ---------------- | --------------------- | ------------------ | ------- |

| 1        | 1                | 128                   | 367.29             | 1.00    |

| 1        | 4                | 128                   | 1449.48            | 3.95    |

| 1        | 8                | 128                   | 2887.65            | 7.86    |

| 2        | 8                | 128                   | 5716.79            | 15.56   |

| 4        | 8                | 128                   | 10917.09           | 29.72   |



### AMP & `dynamic loss scale`

 由于使用 AMP 时，可以选择 `dynamic loss scale` 或者 `static loss scale`，但是不同实现会带来些微（0.8%~4.7%）的性能差异，所以附上两份数据。


- ### ResNet50 v1.5 batch_size = 256


| node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |

| -------- | ---------------- | --------------------- | ------------------ | ------- |

| 1        | 1                | 256                   | 802.9              | 1.00    |

| 1        | 4                | 256                   | 3137.44            | 3.91    |

| 1        | 8                | 256                   | 6154.66            | 7.67    |

| 2        | 8                | 256                   | 11991.94           | 14.94   |

| 4        | 8                | 256                   | 22551.16           | 28.09   |

同时，可支持的 max batch size=256。

### AMP & `static loss scale`

| node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |

| -------- | ---------------- | --------------------- | ------------------ | ------- |

| 1        | 1                | 256                   | 827.86              | 1.00    |

| 1        | 4                | 256                   | 3253.68            | 3.93   |

| 1        | 8                | 256                   |6446.74          | 7.79    |



同时，可支持的 max batch size=256。




NVIDIA的 PyTorch 官方测评结果详见 [ResNet50 v1.5 For PyTorch 的 Results](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/Classification/ConvNets/resnet50v1.5#results)。


Ray 的 PyTorch 官方测评结果详见 [Distributed PyTorch](https://docs.ray.io/en/master/raysgd/raysgd_pytorch.html#benchmarks)。


详细 Log 信息可下载：[ngc_pytorch_resnet50_v1.5.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Pytorch/ngc_pytorch_resnet50_v1.5.tar)。

