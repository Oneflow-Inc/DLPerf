# PyTorch ResNet50 v1.5 测评

## 概述 Overview

本测评在类脑服务器物理机环境下基于PyTorch [/examples](https://github.com/PyTorch/examples/tree/49ec0bd72b85be55579ae8ceb278c66145f593e1) 仓库中提供的 [ResNet50 v1.5](https://github.com/PyTorch/examples/tree/49ec0bd72b85be55579ae8ceb278c66145f593e1/imagenet) 实现，对其进行单机单卡、单机多卡的结果复现及速度评测，同时根据 PyTorch 官方的分布式实现，添加 DALI 数据加载方式，测试 1 机、2 机、4 机的吞吐率及加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试仅覆盖 FP32 精度，后续将持续维护，增加混合精度训练等多种方式的测评。

## 内容目录 Table Of Content

- [概述 Overview](#---overview)
- [内容目录 Table Of Content](#-----table-of-content)
- [环境 Environment](#---environment)
  * [系统](#--)
    + [硬件](#--)
    + [软件](#--)
  * [NGC 容器](#ngc---)
    + [Feature support matrix](#feature-support-matrix)
- [快速开始 Quick Start](#-----quick-start)
  * [1. 前期准备](#1-----)
    + [数据集](#---)
    + [镜像及容器](#-----)
    + [SSH 免密](#ssh---)
  * [2. 运行测试](#2-----)
    + [单机测试](#----)
    + [多机测试](#----)
  * [3. 数据处理](#3-----)
- [性能结果 Performance](#-----performance)
  * [FP32 & W/O XLA & Use `torch.utils.data.DataLoader`](#fp32---w-o-xla---use--torchutilsdatadataloader-)
  * [容器环境内](#-----)
    + [ResNet50 v1.5 batch_size = 128, worker=8](#resnet50-v15-batch-size---128--worker-8)
    + [ResNet50 v1.5 batch_size = 128, worker=48](#resnet50-v15-batch-size---128--worker-48)
  * [物理机环境内](#------)
    + [ResNet50 v1.5 batch_size = 128, worker=8](#resnet50-v15-batch-size---128--worker-8-1)
    + [ResNet50 v1.5 batch_size = 128, worker=48](#resnet50-v15-batch-size---128--worker-48-1)
  * [FP32 & W/O XLA & Use DALI](#fp32---w-o-xla---use-dali)
  * [容器环境内](#------1)
    + [ResNet50 v1.5 batch_size = 128, worker=8](#resnet50-v15-batch-size---128--worker-8-2)
    + [ResNet50 v1.5 batch_size = 128，worker=48](#resnet50-v15-batch-size---128-worker-48)
  * [物理机环境内](#-------1)
    + [ResNet50 v1.5 batch_size = 128, worker=8](#resnet50-v15-batch-size---128--worker-8-3)
    + [ResNet50 v1.5 batch_size = 128, worker=8](#resnet50-v15-batch-size---128--worker-8-4)


## 环境 Environment

### 系统

- #### 硬件

  - GPU：Tesla V100-SXM2-16GB x 8

- #### 软件

  - 驱动：NVIDIA 440.33.01

- 系统：[ Ubuntu 16.04](http://releases.ubuntu.com/16.04/)

  - CUDA：10.2
  - cuDNN：7.6.5
  - NCCL：2.7.3
  - PyTorch：1.6.0

  - OpenMPI：4.0.0
  - DALI：0.19.0
  - Python：3.7.6

#### Feature support matrix

| Feature                                                      | ResNet50 v1.5 PyTorch |
| ------------------------------------------------------------ | --------------------- |
| Multi-gpu training                                           | Yes                   |
| Multi-node training                                          | Yes                   |
| [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) | Yes                   |
| NCCL                                                         | Yes                   |
| Automatic mixed precision (AMP)                              | No                    |



## 快速开始 Quick Start

### 1. 前期准备

- #### 数据集

根据 [Requirements](https://github.com/pytorch/examples/tree/49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de/imagenet#requirements) 准备 ImageNet 数据集，只需下载、解压 train、validation 数据集到对应路径即可，使用原始图片进行训练。

- #### SSH 免密

单机测试下无需配置，但测试 2 机、4 机等多机情况下，则需要配置 docker 容器间的 ssh 免密登录，保证 PyTorch 官方的 mpi/nccl 分布式脚本运行时可以在单机上与其他节点互联。

**安装ssh服务端**

```
# 在容器内执行
apt-get update
apt-get install openssh-server
```

**设置免密登录**

- 节点间的 /root/.ssh/id_rsa.pub 互相授权，添加到 /root/.ssh/authorized_keys 中；
- 修改 sshd 中用于 docker 通信的端口号 `vim /etc/ssh/sshd_config`，修改 `Port` 为空闲端口号；
- 重启 ssh 服务，`service ssh restart`。

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

将本仓库 /DLPerf/PyTorch/resnet50v1.5/scripts 路径源码放至 /examples/imagenet 下，执行脚本

```
bash scripts/run_single_node.sh
```

针对单机单卡、4卡、8卡， batch_size 取 128 等情况分别测试，并将 log 信息保存在当前目录的 /PyTorch/ 对应分布式配置路径中。

> 若测试 DALI 数据加载方式，则需对代码进行修改。

使用 DALI 加载数据可以提升训练效率，增加 image_classification/dataloaders.py

```
import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from functools import partial

DATA_BACKEND_CHOICES = ["pytorch", "syntetic"]
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types

    DATA_BACKEND_CHOICES.append("dali-gpu")
    DATA_BACKEND_CHOICES.append("dali-cpu")
except ImportError:
    print(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )

class HybridTrainPipe(Pipeline):
    def __init__(
        self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False
    ):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=True,
        )

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.ImageDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoder(
                device="mixed",
                output_type=types.RGB,
                device_memory_padding=211025920,
                host_memory_padding=140544512,
            )

        self.res = ops.RandomResizedCrop(
            device=dali_device,
            size=[crop, crop],
            interp_type=types.INTERP_LINEAR,
            random_aspect_ratio=[0.75, 4.0 / 3.0],
            random_area=[0.08, 1.0],
            num_attempts=100,
        )

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]
        
class DALIWrapper(object):
    def gen_wrapper(dalipipeline, num_classes, one_hot, memory_format):
        for data in dalipipeline:
            input = data[0]["data"].contiguous(memory_format=memory_format)
            target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            if one_hot:
                target = expand(num_classes, torch.float, target)
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline, num_classes, one_hot, memory_format):
        self.dalipipeline = dalipipeline
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.memory_format = memory_format

    def __iter__(self):
        return DALIWrapper.gen_wrapper(
                self.dalipipeline, self.num_classes, self.one_hot, self.memory_format
        )

def get_dali_train_loader(dali_cpu=False):
    def gdtl(
        data_path,
        batch_size,
        num_classes,
        one_hot,
        start_epoch=0,
        workers=5,
        _worker_init_fn=None,
        fp16=False,
        memory_format=torch.contiguous_format,
    ):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        traindir = os.path.join(data_path, "train")

        pipe = HybridTrainPipe(
            batch_size=batch_size,
            num_threads=workers,
            device_id=rank % torch.cuda.device_count(),
            data_dir=traindir,
            crop=224,
            dali_cpu=dali_cpu,
        )

        pipe.build()
        train_loader = DALIClassificationIterator(
            pipe, size=int(pipe.epoch_size("Reader") / world_size)
        )

        return (
            DALIWrapper(train_loader, num_classes, one_hot, memory_format),
            int(pipe.epoch_size("Reader") / (world_size * batch_size)),
        )

    return gdtl
```

/examples/imagenet/main.py 也应修改如下：

```
21 from image_classification.dataloaders import *
...
223    # train_loader = torch.utils.data.DataLoader(
224    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
225    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
226     get_train_loader = get_dali_train_loader(dali_cpu=True)
227     args.train_loader_len = 0
228     train_loader, train_loader_len = get_train_loader(
229         args.data,
230         args.batch_size,
231         1000, #args.num_classes,
232         False, #args.mixup > 0.0,
233         start_epoch=args.start_epoch,
234         workers=args.workers,
235         fp16=False, #args.fp16,
236         memory_format=torch.contiguous_format,
237     )
238     args.train_loader_len = train_loader_len
...
285     progress = ProgressMeter(
286         args.train_loader_len, #len(train_loader),
287         [batch_time, data_time, losses, top1, top5],
288         prefix="Epoch: [{}]".format(epoch))
```



- #### 多机测试

  多机测试，一定要确保数据集存在各节点测试机器的相同路径下。脚本默认每轮训练 1 个 epoch，可以直接按照脚本运行，也可自行 Ctrl+C，提前结束。

  1. 两机测试

     以 NODE1 和 NODE2 为例，run_two_nodes.sh 脚本已填入 2 台机器对应的 IP 及端口号，还需自行将 NODE2 机器上相同路径下的脚本 36 行 `CMD+=" --rank 0"` 中的 `rank` 改为 1，在 2 台机器上同时运行脚本，打印 log 如下：

     ```
     root@VS002:~/examples/imagenet/scripts# bash run_two_nodes.sh 
     libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
     libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
     libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
     libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
     libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
     libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
     libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
     libibverbs: Warning: no userspace device-specific driver found for /sys/class/infiniband_verbs/uverbs0
     
     
     
     Use GPU: 6 for training
     => creating model 'resnet50'
     Epoch: [0][ 0/53]       Time 12.054 (12.054)    Data  0.001 ( 0.001)    Loss 6.9828e+00 (6.9828e+00)    Acc@1   0.78 (  0.78) Acc@5   1.56 (  1.56)
     Epoch: [0][ 1/53]       Time  0.970 ( 6.512)    Data  0.002 ( 0.001)    Loss 2.6312e+01 (1.6647e+01)    Acc@1   1.56 (  1.17) Acc@5  10.94 (  6.25)
     Epoch: [0][ 2/53]       Time  0.363 ( 4.462)    Data  0.114 ( 0.039)    Loss 3.3840e+01 (2.2378e+01)    Acc@1   2.34 (  1.56) Acc@5  10.94 (  7.81)
     Epoch: [0][ 3/53]       Time  0.358 ( 3.436)    Data  0.106 ( 0.056)    Loss 1.7229e+01 (2.1091e+01)    Acc@1   3.12 (  1.95) Acc@5  10.16 (  8.40)
     Epoch: [0][ 4/53]       Time  0.352 ( 2.820)    Data  0.104 ( 0.065)    Loss 1.3266e+01 (1.9526e+01)    Acc@1   4.69 (  2.50) Acc@5  17.97 ( 10.31)
     Epoch: [0][ 5/53]       Time  0.360 ( 2.410)    Data  0.108 ( 0.072)    Loss 2.0644e+01 (1.9712e+01)    Acc@1   1.56 (  2.34) Acc@5  13.28 ( 10.81)
     Epoch: [0][ 6/53]       Time  0.403 ( 2.123)    Data  0.106 ( 0.077)    Loss 1.1494e+01 (1.8538e+01)    Acc@1   1.56 (  2.23) Acc@5  11.72 ( 10.94)
     Epoch: [0][ 7/53]       Time  0.555 ( 1.927)    Data  0.107 ( 0.081)    Loss 6.3337e+00 (1.7013e+01)    Acc@1   0.00 (  1.95) Acc@5  13.28 ( 11.23)
     Epoch: [0][ 8/53]       Time  0.529 ( 1.772)    Data  0.107 ( 0.084)    Loss 6.7579e+00 (1.5873e+01)    Acc@1   1.56 (  1.91) Acc@5   9.38 ( 11.02)
     Epoch: [0][ 9/53]       Time  0.437 ( 1.638)    Data  0.107 ( 0.086)    Loss 7.0743e+00 (1.4993e+01)    Acc@1   1.56 (  1.88) Acc@5  11.72 ( 11.09)
     Epoch: [0][10/53]       Time  0.575 ( 1.541)    Data  0.106 ( 0.088)    Loss 7.0139e+00 (1.4268e+01)    Acc@1   2.34 (  1.92) Acc@5   8.59 ( 10.87)
     Epoch: [0][14/53]       Time  0.532 ( 1.261)    Data  0.107 ( 0.093)    Loss 6.7958e+00 (1.2212e+01)    Acc@1   2.34 (  2.03) Acc@5  15.62 ( 11.46)
     Epoch: [0][15/53]       Time  0.508 ( 1.214)    Data  0.107 ( 0.094)    Loss 5.3901e+00 (1.1786e+01)    Acc@1   0.78 (  1.95) Acc@5  16.41 ( 11.77)
     ```

     即为 2 机训练测试成功。

  2. 多机测试

     以本集群为例，最多支持 4 机 32 卡，run_multi_nodes.sh 脚本已设置 NODE1 为 master node，设置好其 IP 及端口号，还需自行将 NODE3 机器上相同路径下的脚本 36 行 `CMD+=" --rank 1"` 中的 `rank` 改为 2， NODE4 的 `rank` 改为 3，在 4 台机器上同时运行脚本，打印 log 如下：

     ```
     Epoch: [0][ 5/13]       Time  0.362 ( 2.620)    Data  0.106 ( 0.075)    Loss 1.4081e+01 (3.0098e+01)    Acc@1   1.56 (  1.95)       Acc@5  14.06 ( 10.42)
     Epoch: [0][ 6/13]       Time  0.403 ( 2.303)    Data  0.149 ( 0.086)    Loss 8.3863e+00 (2.6996e+01)    Acc@1   1.56 (  1.90)       Acc@5   9.38 ( 10.27)
     Epoch: [0][ 7/13]       Time  0.499 ( 2.078)    Data  0.245 ( 0.106)    Loss 8.2231e+00 (2.4650e+01)    Acc@1   2.34 (  1.95)       Acc@5  16.41 ( 11.04)
     Epoch: [0][ 8/13]       Time  0.523 ( 1.905)    Data  0.267 ( 0.124)    Loss 8.7110e+00 (2.2879e+01)    Acc@1   4.69 (  2.26)       Acc@5  10.16 ( 10.94)
     Epoch: [0][ 9/13]       Time  0.501 ( 1.764)    Data  0.227 ( 0.134)    Loss 6.6929e+00 (2.1260e+01)    Acc@1   2.34 (  2.27)       Acc@5  15.62 ( 11.41)
     Epoch: [0][10/13]       Time  0.480 ( 1.648)    Data  0.194 ( 0.139)    Loss 6.4846e+00 (1.9917e+01)    Acc@1   1.56 (  2.20)       Acc@5  10.94 ( 11.36)
     Epoch: [0][11/13]       Time  0.497 ( 1.552)    Data  0.233 ( 0.147)    Loss 6.5990e+00 (1.8807e+01)    Acc@1   3.12 (  2.28)       Acc@5  11.72 ( 11.39)
     Epoch: [0][12/13]       Time  0.438 ( 1.466)    Data  0.157 ( 0.148)    Loss 6.2312e+00 (1.7840e+01)    Acc@1   4.69 (  2.46)       Acc@5  13.28 ( 11.54)
     Epoch: [0][13/13]       Time  0.548 ( 1.401)    Data  0.268 ( 0.157)    Loss 6.5306e+00 (1.7032e+01)    Acc@1   2.34 (  2.46)       Acc@5  10.16 ( 11.44)
     Use GPU: 1 for training
     => creating model 'resnet50'
     Epoch: [0][ 0/13]       Time 13.508 (13.508)    Data  0.009 ( 0.009)    Loss 7.0640e+00 (7.0640e+00)    Acc@1   0.00 (  0.00)       Acc@5   0.00 (  0.00)
     Epoch: [0][ 1/13]       Time  0.774 ( 7.141)    Data  0.003 ( 0.006)    Loss 1.9974e+01 (1.3519e+01)    Acc@1   2.34 (  1.17)       Acc@5  12.50 (  6.25)
     Epoch: [0][ 2/13]       Time  0.317 ( 4.866)    Data  0.066 ( 0.026)    Loss 2.6350e+01 (1.7796e+01)    Acc@1   1.56 (  1.30)       Acc@5  11.72 (  8.07)
     Epoch: [0][ 3/13]       Time  0.361 ( 3.740)    Data  0.110 ( 0.047)    Loss 3.1926e+01 (2.1329e+01)    Acc@1   4.69 (  2.15)       Acc@5  18.75 ( 10.74)
     Epoch: [0][ 4/13]       Time  0.359 ( 3.064)    Data  0.108 ( 0.059)    Loss 1.1616e+01 (1.9386e+01)    Acc@1   1.56 (  2.03)       Acc@5  10.16 ( 10.62)
     Epoch: [0][ 5/13]       Time  0.363 ( 2.614)    Data  0.108 ( 0.067)    Loss 9.9896e+00 (1.7820e+01)    Acc@1   0.78 (  1.82)       Acc@5   8.59 ( 10.29)
     ```

     即为 4 机训练测试成功。

### 3. 数据处理

测试进行了多组训练（本测试中取 5 次），由于多卡训练时日志异步打印，每张卡第 1 个 iter 的处理时间是其他 iter 的几何倍，导致最后的平均值会产生较大误差，因此考虑将每张卡运行结果的前 5 iter 舍去。每次训练过程取第 1 个 epoch 的舍去 25 iter 后的数据，计算训练速度时只取后 100 iter 的数据，以降低抖动。最后将 5 次训练的结果取中位数得到最终速度，并最终以此数据计算加速比。

运行 /DLPerf/PyTorch/resnet50v1.5/extract_PyTorch_logs_time.py，即可得到针对不同配置测试结果 log 数据处理的结果： 

```
python extract_pytorch_logs_time.py --log_dir /root/examples/imagenet/scripts/j48_pytorch_original --warmup_batches 20 --train_batches 120 --batch_size_per_device 128
```

结果打印如下

```
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_42.log {42: 8940.3}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_2.log {42: 8940.3, 2: 8468.23}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_4.log {42: 8940.3, 2: 8468.23, 4: 8781.78}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_3.log {42: 8940.3, 2: 8468.23, 4: 8781.78, 3: 8648.47}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_52.log {42: 8940.3, 2: 8468.23, 4: 8781.78, 3: 8648.47, 52: 8851.62}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_32.log {42: 8940.3, 2: 8468.23, 4: 8781.78, 3: 8648.47, 52: 8851.62, 32: 8654.31}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_5.log {42: 8940.3, 2: 8468.23, 4: 8781.78, 3: 8648.47, 52: 8851.62, 32: 8654.31, 5: 9046.34}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_12.log {42: 8940.3, 2: 8468.23, 4: 8781.78, 3: 8648.47, 52: 8851.62, 32: 8654.31, 5: 9046.34, 12: 8973.6}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_22.log {42: 8940.3, 2: 8468.23, 4: 8781.78, 3: 8648.47, 52: 8851.62, 32: 8654.31, 5: 9046.34, 12: 8973.6, 22: 8515.95}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/4n8g/r50_b128_fp32_1.log {42: 8940.3, 2: 8468.23, 4: 8781.78, 3: 8648.47, 52: 8851.62, 32: 8654.31, 5: 9046.34, 12: 8973.6, 22: 8515.95, 1: 8853.15}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n1g/r50_b128_fp32_2.log {2: 353.62}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n1g/r50_b128_fp32_4.log {2: 353.62, 4: 354.44}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n1g/r50_b128_fp32_3.log {2: 353.62, 4: 354.44, 3: 354.02}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n1g/r50_b128_fp32_5.log {2: 353.62, 4: 354.44, 3: 354.02, 5: 354.47}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n1g/r50_b128_fp32_1.log {2: 353.62, 4: 354.44, 3: 354.02, 5: 354.47, 1: 354.61}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/2n8g/r50_b128_fp32_2.log {2: 4484.64}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/2n8g/r50_b128_fp32_4.log {2: 4484.64, 4: 4401.46}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/2n8g/r50_b128_fp32_3.log {2: 4484.64, 4: 4401.46, 3: 4347.27}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/2n8g/r50_b128_fp32_5.log {2: 4484.64, 4: 4401.46, 3: 4347.27, 5: 4372.33}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/2n8g/r50_b128_fp32_1.log {2: 4484.64, 4: 4401.46, 3: 4347.27, 5: 4372.33, 1: 4038.81}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n8g/r50_b128_fp32_2.log {2: 2277.07}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n8g/r50_b128_fp32_4.log {2: 2277.07, 4: 2311.41}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n8g/r50_b128_fp32_3.log {2: 2277.07, 4: 2311.41, 3: 2244.24}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n8g/r50_b128_fp32_5.log {2: 2277.07, 4: 2311.41, 3: 2244.24, 5: 2243.7}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n8g/r50_b128_fp32_1.log {2: 2277.07, 4: 2311.41, 3: 2244.24, 5: 2243.7, 1: 2185.75}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n4g/r50_b128_fp32_2.log {2: 1367.67}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n4g/r50_b128_fp32_4.log {2: 1367.67, 4: 1336.61}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n4g/r50_b128_fp32_3.log {2: 1367.67, 4: 1336.61, 3: 1367.89}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n4g/r50_b128_fp32_5.log {2: 1367.67, 4: 1336.61, 3: 1367.89, 5: 1340.95}
/home/leinao/examples/imagenet/scripts/physical_j8_dali/1n4g/r50_b128_fp32_1.log {2: 1367.67, 4: 1336.61, 3: 1367.89, 5: 1340.95, 1: 1369.97}
{'r50': {'1n1g': {'average_speed': 354.23,
                  'batch_size_per_device': 128,
                  'median_speed': 354.44,
                  'speedup': 1.0},
         '1n4g': {'average_speed': 1356.62,
                  'batch_size_per_device': 128,
                  'median_speed': 1367.67,
                  'speedup': 3.86},
         '1n8g': {'average_speed': 2252.43,
                  'batch_size_per_device': 128,
                  'median_speed': 2244.24,
                  'speedup': 6.33},
         '2n8g': {'average_speed': 4328.9,
                  'batch_size_per_device': 128,
                  'median_speed': 4372.33,
                  'speedup': 12.34},
         '4n8g': {'average_speed': 8773.38,
                  'batch_size_per_device': 128,
                  'median_speed': 8816.7,
                  'speedup': 24.88}}}
Saving result to ./result/_result.json
```



## 性能结果 Performance

该小节提供针对 NVIDIA PyTorch 框架的 ResNet50 v1.5 模型测试的性能结果和完整 log 日志。

### FP32 & W/O XLA & Use `torch.utils.data.DataLoader`

- ### 容器环境内

  - #### ResNet50 v1.5 batch_size = 128, worker=8

  | node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
  | -------- | ---------------- | --------------------- | ------------------ | ------- |
  | 1        | 1                | 128                   | 354.81             | 1.00    |
  | 1        | 4                | 128                   | 1330.25            | 3.75    |
  | 1        | 8                | 128                   | 1630.24            | 4.59    |
  | 2        | 8                | 128                   | 3211.04            | 9.05    |
  | 4        | 8                | 128                   | 6410.82            | 18.07   |

  

  - #### ResNet50 v1.5 batch_size = 128, worker=48

  | node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
  | -------- | ---------------- | --------------------- | ------------------ | ------- |
  | 1        | 1                | 128                   | 354.4              | 1.00    |
  | 1        | 4                | 128                   | 1350.96            | 3.81    |
  | 1        | 8                | 128                   | 2719.07            | 7.67    |
  | 2        | 8                | 128                   | 5307.21            | 14.98   |
  | 4        | 8                | 128                   | 10632.33           | 30.0    |

- ### 物理机环境内

  - #### ResNet50 v1.5 batch_size = 128, worker=8

  | node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
  | -------- | ---------------- | --------------------- | ------------------ | ------- |
  | 1        | 1                | 128                   | 306.35             | 1.00    |
  | 1        | 4                | 128                   | 1079.19            | 3.52    |
  | 1        | 8                | 128                   | 1070.37            | 3.49    |
  | 2        | 8                | 128                   | 2093.53            | 6.83    |
  | 4        | 8                | 128                   | 4119.84            | 13.45   |

  - #### ResNet50 v1.5 batch_size = 128, worker=48

  | node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
  | -------- | ---------------- | --------------------- | ------------------ | ------- |
  | 1        | 1                | 128                   | 348.62             | 1.00    |
  | 1        | 4                | 128                   | 1226.38            | 3.52    |
  | 1        | 8                | 128                   | 2632.93            | 7.55    |
  | 2        | 8                | 128                   | 5115.4             | 14.67   |
  | 4        | 8                | 128                   | 10021.29           | 28.75   |

### FP32 & W/O XLA & Use DALI

- ### 容器环境内

  - #### ResNet50 v1.5 batch_size = 128, worker=8

  | node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
  | -------- | ---------------- | --------------------- | ------------------ | ------- |
  | 1        | 1                | 128                   | 361.16             | 1.00    |
  | 1        | 4                | 128                   | 1314.3             | 3.64    |
  | 1        | 8                | 128                   | 2171.01            | 6.01    |
  | 2        | 8                | 128                   | 4221.2             | 11.69   |
  | 4        | 8                | 128                   | 8151.08            | 22.57   |

  - #### ResNet50 v1.5 batch_size = 128，worker=48

  | node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
  | -------- | ---------------- | --------------------- | ------------------ | ------- |
  | 1        | 1                | 128                   | 357.91             | 1.00    |
  | 1        | 4                | 128                   | 1273.76            | 3.56    |
  | 1        | 8                | 128                   | 2736.8             | 7.65    |
  | 2        | 8                | 128                   | 5251.01            | 14.67   |
  | 4        | 8                | 128                   | 10297.15           | 28.77   |

- ### 物理机环境内

  - #### ResNet50 v1.5 batch_size = 128, worker=8

  | node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
  | -------- | ---------------- | --------------------- | ------------------ | ------- |
  | 1        | 1                | 128                   | 354.44             | 1.00    |
  | 1        | 4                | 128                   | 1367.67            | 3.86    |
  | 1        | 8                | 128                   | 2244.24            | 6.33    |
  | 2        | 8                | 128                   | 4372.33            | 12.34   |
  | 4        | 8                | 128                   | 8816.7             | 24.88   |

  - #### ResNet50 v1.5 batch_size = 128, worker=48

  | node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
  | -------- | ---------------- | --------------------- | ------------------ | ------- |
  | 1        | 1                | 128                   | 350.66             | 1.00    |
  | 1        | 4                | 128                   | 1306.49            | 3.73    |
  | 1        | 8                | 128                   | 2707.42            | 7.72    |
  | 2        | 8                | 128                   | 5193.09            | 14.81   |
  | 4        | 8                | 128                   | 10032.72           | 28.61   |



NVIDIA的 PyTorch 官方测评结果详见 [ResNet50 v1.5 For PyTorch 的 results](https://github.com/NVIDIA/DeepLearningExamples/blob/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/LanguageModeling/BERT/README.md#results)。

Ray 的 PyTorch 官方测评结果详见 [Distributed PyTorch](https://docs.ray.io/en/master/raysgd/raysgd_PyTorch.html#benchmarks).

详细 Log 信息可下载：[PyTorch_example_resnet50_v1.5.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/PyTorch/pytorch_example_resnet50_v1.5.tar)
