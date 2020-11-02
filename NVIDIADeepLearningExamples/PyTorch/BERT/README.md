# NVIDIA/DeepLearningExamples PyTorch BERT 测评

## 概述 Overview

本测试基于 [NVIDIA/DeepLearningExamples/PyTorch/LanguageModeling/BERT/](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4) 仓库中提供的 PyTorch 框架的 BERT 实现，在 NVIDIA 官方提供的 [20.03 NGC 镜像及其衍生容器](https://ngc.nvidia.com/catalog/containers/nvidia:PyTorch/tags)中进行单机单卡、单机多卡、多机多卡的结果复现及速度评测，同时增加分布式实现，测试 1 机、2 机、4 机的吞吐率及加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试覆盖 FP32 及混合精度，后续将持续维护，增加使用其他优化方式的测评。

## 内容目录 Table Of Contents

- [NVIDIA/DeepLearningExamples PyTorch BERT 测评](#nvidia-deeplearningexamples-pytorch-bert---)
  * [概述 Overview](#---overview)
  * [内容目录 Table Of Contents](#-----table-of-contents)
  * [环境 Environment](#---environment)
    + [系统](#--)
      - [硬件](#--)
      - [软件](#--)
    + [NGC 容器](#ngc---)
      * [Feature support matrix](#feature-support-matrix)
  * [快速开始 Quick Start](#-----quick-start)
    + [1. 前期准备](#1-----)
      - [数据集](#---)
      - [镜像及容器](#-----)
      - [SSH 免密](#ssh---)
      - [Adam 算法](#adam---)
    + [2. 运行测试](#2-----)
      - [单机测试](#----)
      - [多机测试](#----)
    + [3. 数据处理](#3-----)
  * [性能结果 Performance](#-----performance)
    + [FP32](#fp32)
      - [BERT-Base batch_size = 32](#bert-base-batch-size---32)
      - [BERT-Base batch_size = 48](#bert-base-batch-size---48)
  * [FP16](#fp16)
    - [BERT-Base batch_size = 64](#bert-base-batch-size---64)
    - [BERT-Base batch_size = 96](#bert-base-batch-size---96)

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

  ##### Feature support matrix

  | Feature                         | BERT PyTorch |
  | ------------------------------- | ------------ |
  | Multi-gpu training              | Yes          |
  | Multi-node                      | Yes          |
  | Automatic mixed precision (AMP) | Yes          |
  | NVIDIA NCCL                     | Yes          |



## 快速开始 Quick Start

### 1. 前期准备

- #### 数据集

根据 [BERT For PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/LanguageModeling/BERT#bert-for-pytorch) 中的 [Getting the data](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/LanguageModeling/BERT#getting-the-data) 小节准备 PyTorch 使用的 `hdf5` 格式的 BERT 数据集，主要有 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (fine-tuning for question answering) 、Wikipedia (pre-training)、BookCorpus (pre-training)。

考虑到性能测试无需花费大量时间（网络良好，梯子健全情况下大约一天）制备完整数据集，简易 Wikipedia 数据集制作可参考以下步骤：

- 下载 Wikipedia 数据集并解压，取其 /AA、/AB 等节选数据路径数据作为使用的 data sample，放至 /workspace/examples/bert/data/extracted 下。

- 修改 /workspace/examples/bert/data/WikicorpusTextFormatting.py，如下

  ```
  import glob
  import os
  import argparse
  
  class WikicorpusTextFormatting:
      def __init__(self, args, recursive = False):
          self.wiki_path = args.wiki_path
          self.recursive = recursive
          self.output_filename = args.output_filename
  
  
      # This puts one article per line
      def merge(self):
          print("wiki_path: ", self.wiki_path)
          print("output_filename: ", self.output_filename)
          with open(self.output_filename, mode='w', newline='\n') as ofile:
              for dirname in glob.glob(self.wiki_path + '/*/', recursive=False):
                  for filename in glob.glob(dirname + 'wiki_*', recursive=self.recursive):
                      print(filename)
                      article_lines = []
                      article_open = False
  
                      with open(filename, mode='r', newline='\n') as file:
                          for line in file:
                              if '<doc id=' in line:
                                  article_open = True
                              elif '</doc>' in line:
                                  article_open = False
                                  for oline in article_lines[1:]:
                                      if oline != '\n':
                                          ofile.write(oline.rstrip() + " ")
                                  ofile.write("\n\n")
                                  article_lines = []
                              else:
                                  if article_open:
                                      article_lines.append(line)
  
  if __name__ == "__main__":
      parser = argparse.ArgumentParser(description='Preprocessing Wiki Corpus...')
  
      parser.add_argument("--wiki_path", type=str, default="/workspace/bert/data/extracted", help="input wiki path")
      parser.add_argument("--output_filename", type=str, default="/workspace/bert/data/formatted_one_article_per_line/wikicorpus_en_one_book_per_line.txt", help="output file name")
      args = parser.parse_args()
      wiki_corpus = WikicorpusTextFormatting(args, recursive=True)
      wiki_corpus.merge()
      print("merge done.")
  ```

  执行成功，会生成 data/formatted_one_article_per_line/wikicorpus_en_one_book_per_line.txt 文件。

  然后，注释掉 /workspace/examples/bert/data/create_datasets_from_start.sh 中的 40 行之前 `Download` 和  `Properly format the text files` 相关代码，直接进行 `Shard the text files` 和 `create HDF5 files PHase 1` 和 `Create HDF5 files Phase 2` 即可。

  可以先制作数据集，运行容器时绑定数据集路径（`-v ./data:/data/`），也可以先起容器，制作完数据集，使用 scp 传输数据集至容器内的路径，并修改脚本中的数据集路径。

- #### 镜像及容器

拉取 NGC 20.03 的镜像、搭建容器，进入容器环境。

```
# 下载镜像
docker pull nvcr.io/nvidia/pytorch:20.03-py3 

# 启动容器
docker run -it --shm-size=16g --ulimit memlock=-1 --privileged  \
--name pt_bert  --net host \
--cap-add=IPC_LOCK --device=/dev/infiniband \
-v ./data:/data/ \
-d pytorch:20.03-py3 
```

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



- ## 安装 IB 驱动 ##

测试机器上的容器环境内未查找到 IB 驱动，会导致测试时 NCCL 库只能使用 Socket 通信，无法达到最佳测试效果，因此需要额外安装，具体可参考[安装 IB 驱动](https://github.com/Oneflow-Inc/DLPerf/tree/master/NVIDIADeepLearningExamples/PyTorch/resnet50v1.5#%E5%AE%89%E8%A3%85-ib-%E9%A9%B1%E5%8A%A8)。

- #### Adam 算法

为了保持算法一致性，脚本 /workspace/examples/bert/run_pretraining.py 中的 `FusedLAMB` 被替换为 `FusedAdam`，该脚本需做如下修改：

```
43 from apex.optimizers import `FusedLAMB, FusedAdam # add FusedAdam
.....
# modify FusedLAMB with FusedAdam
322     optimizer = FusedAdam(optimizer_grouped_parameters,
323                           lr=args.learning_rate,
324                           bias_correction=False)
```

如此即可。

### 2. 运行测试

本次测试集群中有 4 台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有 8 张 V100 显卡， 每张显卡显存 16 G。

- #### 单机测试

在容器内下载本仓库源码：

````
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 /DLPerf/NVIDIADeepLearningExamples/PyTorch/BERT/scripts 目录源码移至 /workspace/examples/bert/test_scripts（需新建） 下，执行脚本

```
bash run_single_node.sh
```

即可执行针对单机单卡、单机 2 卡、4 卡、 8 卡， batch_size 分别取 32、48 等情况的集成测试，并将 log 信息保存在当前目录的 /ngc/pytorch/ 对应分布式配置路径中，如单机单卡为 /1n1g，意为 1 node 1 gpu；单机 8卡为 /1n8g，意为 1 node 8 gpus，以此类推。

如需测试 `amp`，修改脚本中的 `PREC` 为 `amp` 即可。

若想测试 `fp16`，直接使用 `--fp16` 选项，会提示

```
Warning:  FP16_Optimizer is deprecated and dangerous, and will be deleted soon.  
If it still works, you're probably getting lucky.  
For mixed precision, use the documented API https://nvidia.github.io/apex/amp.html, 
with opt_level=O1.
```

因此建议使用 `amp` 参数或者在验证过正确性情况下修改源码中的 `opt_level` 。

- #### 多机测试

多机测试，一定要确保数据集存在各节点测试机器的相同路径下，各脚本的行为要一致，尤其是修改要保持同步。

如需测试 `amp`，直接修改脚本中的 `PREC` 为 `amp` 即可。

- **两机测试**

以 NODE1 和 NODE2 为例，run_two_nodes.sh 脚本已填入 2 台机器对应的 IP 及端口号，NODE1 上的脚本 single_node_train.sh 中 `--node_rank` 默认为 0，还需自行将 NODE2 机器上相同路径下的脚本 108 行 `--node_rank` 改为 1，在 2 台机器上同时运行脚本，

```
bash run_two_node.sh
```

- **多机测试**

以本集群为例，最多支持 4 机 32 卡，run_multi_nodes.sh 脚本已设置 NODE1 为 master node，设置好其 IP 及端口号，还需自行将 NODE3 机器上相同路径下的脚本 108 行 `--node_rank` 中的改为 2， NODE4 的 `--node_rank` 改为 3，在 4 台机器上同时运行脚本，

```
bash run_multi_nodes.sh
```

即可执行多节点 batch_size 分别取 32、48 等情况的集成测试，并将 log 信息保存在当前目录的对应分布式配置路径中。

### 3. 数据处理

测试进行了多组训练（本测试中取 5 次），每次训练过程只取第 1 个 epoch 的前 150 iter，计算训练速度时取后 100 iter 的数据，以降低抖动。最后将 5 次训练的结果取中位数得到最终速度，并以此数据计算加速比。

运行 /DLPerf/NVIDIADeepLearningExamples/PyTorch/BERT/extract_pytorch_logs_time.py，即可得到针对不同配置测试结果 log 数据处理的结果： 

```
python extract_pytorch_logs_time.py --log_dir /workspace/examples/bert/test_scripts/ngc_bert_b48/pytorch --warmup_batches 20 --train_batches 120 --batch_size_per_device 32
```

结果打印如下

```
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/4n8g/bert-base-adam-training_b96_fp16_5.log {5: 10273.14}
end_time:  2020-09-24 02:12:08.999291
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/4n8g/bert-base-adam-training_b96_fp16_1.log {5: 10273.14, 1: 10552.87}
end_time:  2020-09-24 02:15:18.098056
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/4n8g/bert-base-adam-training_b96_fp16_3.log {5: 10273.14, 1: 10552.87, 3: 10324.68}
end_time:  2020-09-24 02:16:52.945844
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/4n8g/bert-base-adam-training_b96_fp16_4.log {5: 10273.14, 1: 10552.87, 3: 10324.68, 4: 10349.12}
end_time:  2020-09-24 02:13:43.531300
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/4n8g/bert-base-adam-training_b96_fp16_2.log {5: 10273.14, 1: 10552.87, 3: 10324.68, 4: 10349.12, 2: 10414.77}
end_time:  2020-09-24 03:20:44.972941
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n1g/bert-base-adam-training_b96_fp16_5.log {5: 463.85}
end_time:  2020-09-24 03:15:44.213131
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n1g/bert-base-adam-training_b96_fp16_1.log {5: 463.85, 1: 462.35}
end_time:  2020-09-24 03:18:14.318222
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n1g/bert-base-adam-training_b96_fp16_3.log {5: 463.85, 1: 462.35, 3: 466.94}
end_time:  2020-09-24 03:19:29.565003
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n1g/bert-base-adam-training_b96_fp16_4.log {5: 463.85, 1: 462.35, 3: 466.94, 4: 462.14}
end_time:  2020-09-24 03:16:58.796182
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n1g/bert-base-adam-training_b96_fp16_2.log {5: 463.85, 1: 462.35, 3: 466.94, 4: 462.14, 2: 462.35}
end_time:  2020-09-24 02:26:50.557894
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/2n8g/bert-base-adam-training_b96_fp16_5.log {5: 5366.7}
end_time:  2020-09-24 02:20:55.793547
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/2n8g/bert-base-adam-training_b96_fp16_1.log {5: 5366.7, 1: 5426.07}
end_time:  2020-09-24 02:23:47.979051
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/2n8g/bert-base-adam-training_b96_fp16_3.log {5: 5366.7, 1: 5426.07, 3: 5448.97}
end_time:  2020-09-24 02:25:18.862542
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/2n8g/bert-base-adam-training_b96_fp16_4.log {5: 5366.7, 1: 5426.07, 3: 5448.97, 4: 5439.94}
end_time:  2020-09-24 02:22:21.485900
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/2n8g/bert-base-adam-training_b96_fp16_2.log {5: 5366.7, 1: 5426.07, 3: 5448.97, 4: 5439.94, 2: 5410.84}
end_time:  2020-09-24 03:34:57.059096
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n8g/bert-base-adam-training_b96_fp16_5.log {5: 3339.15}
end_time:  2020-09-24 03:29:01.089925
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n8g/bert-base-adam-training_b96_fp16_1.log {5: 3339.15, 1: 3260.58}
end_time:  2020-09-24 03:31:53.242659
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n8g/bert-base-adam-training_b96_fp16_3.log {5: 3339.15, 1: 3260.58, 3: 3260.74}
end_time:  2020-09-24 03:33:31.687091
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n8g/bert-base-adam-training_b96_fp16_4.log {5: 3339.15, 1: 3260.58, 3: 3260.74, 4: 3310.51}
end_time:  2020-09-24 03:30:27.478401
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n8g/bert-base-adam-training_b96_fp16_2.log {5: 3339.15, 1: 3260.58, 3: 3260.74, 4: 3310.51, 2: 3287.12}
end_time:  2020-09-24 03:27:35.906235
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n4g/bert-base-adam-training_b96_fp16_5.log {5: 1727.93}
end_time:  2020-09-24 03:22:04.678285
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n4g/bert-base-adam-training_b96_fp16_1.log {5: 1727.93, 1: 1734.78}
end_time:  2020-09-24 03:24:55.125125
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n4g/bert-base-adam-training_b96_fp16_3.log {5: 1727.93, 1: 1734.78, 3: 1731.19}
end_time:  2020-09-24 03:26:14.931147
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n4g/bert-base-adam-training_b96_fp16_4.log {5: 1727.93, 1: 1734.78, 3: 1731.19, 4: 1726.71}
end_time:  2020-09-24 03:23:23.394265
/workspace/examples/bert/test_scripts/fp16_ngc_bert_b96/pytorch/1n4g/bert-base-adam-training_b96_fp16_2.log {5: 1727.93, 1: 1734.78, 3: 1731.19, 4: 1726.71, 2: 1723.52}
{'bert-base-adam-training': {'1n1g': {'average_speed': 463.53,
                                      'batch_size_per_device': 96,
                                      'median_speed': 462.35,
                                      'speedup': 1.0},
                             '1n4g': {'average_speed': 1728.83,
                                      'batch_size_per_device': 96,
                                      'median_speed': 1727.93,
                                      'speedup': 3.74},
                             '1n8g': {'average_speed': 3291.62,
                                      'batch_size_per_device': 96,
                                      'median_speed': 3287.12,
                                      'speedup': 7.11},
                             '2n8g': {'average_speed': 5418.5,
                                      'batch_size_per_device': 96,
                                      'median_speed': 5426.07,
                                      'speedup': 11.74},
                             '4n8g': {'average_speed': 10382.92,
                                      'batch_size_per_device': 96,
                                      'median_speed': 10349.12,
                                      'speedup': 22.38}}}
Saving result to ./result/_result.json
```

## 性能结果 Performance

该小节提供针对 NVIDIA PyTorch 框架的 BERT 模型测试的性能结果和完整 log 日志。

### FP32 

- #### BERT-Base batch_size = 32

| node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 32                    | 119.6              | 1.00    |
| 1        | 4                | 32                    | 457.72             | 3.83    |
| 1        | 8                | 32                    | 921.32             | 7.7     |
| 2        | 8                | 32                    | 1499.4             | 12.54   |
| 4        | 8                | 32                    | 2885.81            | 24.13   |

- #### BERT-Base batch_size = 48

| node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 48                    | 121.94             | 1.00    |
| 1        | 4                | 48                    | 464.66             | 3.81    |
| 1        | 8                | 48                    | 928.01             | 7.61    |
| 2        | 8                | 48                    | 1584.32            | 12.99   |
| 4        | 8                | 48                    | 3039.3             | 24.92   |

## FP16 

- #### BERT-Base batch_size = 64

| node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 64                    | 444.51             | 1.0     |
| 1        | 4                | 64                    | 1671.66            | 3.76    |
| 1        | 8                | 64                    | 3251.7             | 7.32    |
| 2        | 8                | 64                    | 4936.92            | 11.11   |
| 4        | 8                | 64                    | 9331.72            | 20.99   |



- #### BERT-Base batch_size = 96

| node_num | gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
| -------- | ---------------- | --------------------- | ------------------ | ------- |
| 1        | 1                | 96                    | 462.35             | 1.0     |
| 1        | 4                | 96                    | 1727.93            | 3.74    |
| 1        | 8                | 96                    | 3287.12            | 7.11    |
| 2        | 8                | 96                    | 5426.07            | 11.74   |
| 4        | 8                | 96                    | 10349.12           | 22.38   |

同时，可支持的 max batch size=96。

NVIDIA的 PyTorch 官方测评结果详见 [BERT For PyTorch - Performance Results](https://github.com/NVIDIA/DeepLearningExamples/blob/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/LanguageModeling/BERT/README.md#results)

详细 Log 信息可下载：[ngc_pytorch_bert.tar](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/NVIDIA/Pytorch/ngc_pytorch_bert.tar)
