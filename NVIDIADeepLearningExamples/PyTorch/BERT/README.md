# NVIDIA/DeepLearningExamples PyTorch BERT 测评

## 概述 Overview

本测试基于 [NVIDIA/DeepLearningExamples/PyTorch/LanguageModeling/BERT/](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) 仓库中提供的 PyTorch 框架的 BERT 实现，在 NVIDIA 官方提供的 [20.03 NGC 镜像及其衍生容器](https://ngc.nvidia.com/catalog/containers/nvidia:PyTorch/tags)中进行单机单卡、单机多卡的结果复现及速度评测，同时增加分布式实现，测试 1 机、2 机、4 机的吞吐率及加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试仅覆盖 FP32 精度，后续将持续维护，增加混合精度训练，XLA 等多种方式的测评。

## 内容目录 Table Of Contents

- [NVIDIA/DeepLearningExamples PyTorch BERT 测评](#nvidia-deeplearningexamples-PyTorch-bert---)
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
    + [2. 运行测试](#2-----)
    + [3. 数据处理](#3-----)
  * [性能结果 Performance](#-----performance)
    + [FP32 & W/O XLA](#fp32---w-o-xla)
      - [BERT-Base batch_size = 32](#bert-base-batch-size---32)
      - [BERT-Base batch_size = 48](#bert-base-batch-size---48)


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
  | Automatic mixed precision (AMP) | No           |



## 快速开始 Quick Start

### 1. 前期准备

- #### 数据集

根据 [BERT For PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) 中的 [Getting the data](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data) 小节准备 PyTorch 使用的 `hdf5` 格式的 BERT 数据集，主要有 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (fine-tuning for question answering) 、Wikipedia (pre-training)、BookCorpus (pre-training)。

考虑到性能测试无需花费大量时间（网络良好，梯子健全情况下大约一天）制备完整数据集，简易 Wikipedia 数据集制作可参考以下步骤：

- 下载 Wikipedia 数据集并解压，取其 /AA、/AB 等节选数据集路径数据作为使用的 data sample，放至 /workspace/examples/bert/data/extracted 下。

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

  执行成功，会生成 data/formatted_one_article_per_line/wikicorpus_en_one_book_per_line.txt 文件

  然后注释掉 /workspace/examples/bert/data/create_datasets_from_start.sh 中的 40 行之前`Download` 和  `Properly format the text files` 相关代码，直接进行 `Shard the text files` 和 `create HDF5 files PHase 1` 和 `Create HDF5 files Phase 2` 即可。

  可以先制作数据集，运行容器时绑定数据集路径（`-v ./data:/data/`），也可以先起容器，制作完数据集，使用 scp 传递数据集至容器内的路径，并修改脚本中的数据集路径。

- #### 镜像及容器

同时，根据 [NVIDIA 官方指导 Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide)下载源码、拉取镜像（本次测试选用的是 NGC 20.03）、搭建容器，进入容器环境。

```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/LanguageModeling/BERT

# 构建项目镜像 
# DeepLearningExamples/PyTorch/LanguageModeling/BERT目录下
docker build . -t nvidia_rn50_pt:20.03-resnet
# 启动容器
docker  run -it --shm-size=16g --ulimit memlock=-1 --privileged  \
--name pt_bert  --net host \
--cap-add=IPC_LOCK --device=/dev/infiniband \
-v ./data:/data/ \
-d nvidia_rn50_pt:20.03
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

### 2. 运行测试

本次测试集群中有 4 台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有 8 张 V100 显卡， 每张显卡显存 16 G。

- **单机测试**

在容器内下载本仓库源码：

````
git clone https://github.com/Oneflow-Inc/DLPerf.git
````

将本仓库 /DLPerf/NVIDIADeepLearningExamples/PyTorch/BERT/scripts 目录源码移至 /workspace/examples/bert/test_scripts（需新建） 下，执行脚本

```
bash run_single_node.sh
```

即可执行针对单机单卡、单机 2 卡、单机 4 卡、单机 8 卡， batch_size 分别取 32、48 等情况的集成测试，并将 log 信息保存在当前目录的 /ngc/PyTorch/ 对应分布式配置路径中，如单机单卡为 /1n1g，意为 1 node 1 gpu；单机 8卡 为 /1n8g，意为 1 node 8 gpus，以此类推。

### 3. 数据处理

测试进行了多组训练（本测试中取 5 次），每次训练过程只取第 1 个 epoch 的前 120 iter，计算训练速度时只取后 100 iter 的数据，以降低抖动。最后将 5~7 次训练的速度取中位数得到最终速度，并最终以此数据计算加速比。

运行 /DLPerf/NVIDIADeepLearningExamples/PyTorch/BERT/extract_PyTorch_logs_time.py，即可得到针对不同配置测试结果 log 数据处理的结果： 

```
python extract_pytorch_logs_time.py --log_dir [log_dir] --warmup_batches 20 --train_batches 120 --batch_size_per_device 32
```

结果打印如下

```
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n2g/bert-base-adam-training_b32_fp32_1.log {1: 221.39}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n2g/bert-base-adam-training_b32_fp32_5.log {1: 221.39, 5: 221.01}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n2g/bert-base-adam-training_b32_fp32_2.log {1: 221.39, 5: 221.01, 2: 221.13}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n2g/bert-base-adam-training_b32_fp32_3.log {1: 221.39, 5: 221.01, 2: 221.13, 3: 221.18}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n2g/bert-base-adam-training_b32_fp32_4.log {1: 221.39, 5: 221.01, 2: 221.13, 3: 221.18, 4: 221.49}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n1g/bert-base-adam-training_b32_fp32_1.log {1: 120.01}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n1g/bert-base-adam-training_b32_fp32_5.log {1: 120.01, 5: 119.49}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n1g/bert-base-adam-training_b32_fp32_2.log {1: 120.01, 5: 119.49, 2: 119.72}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n1g/bert-base-adam-training_b48_fp32_1.log {1: 0, 5: 119.49, 2: 119.72}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n1g/bert-base-adam-training_b32_fp32_3.log {1: 0, 5: 119.49, 2: 119.72, 3: 119.71}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n1g/bert-base-adam-training_b32_fp32_4.log {1: 0, 5: 119.49, 2: 119.72, 3: 119.71, 4: 119.61}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n8g/bert-base-adam-training_b32_fp32_1.log {1: 909.46}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n8g/bert-base-adam-training_b32_fp32_5.log {1: 909.46, 5: 909.68}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n8g/bert-base-adam-training_b32_fp32_2.log {1: 909.46, 5: 909.68, 2: 908.85}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n8g/bert-base-adam-training_b32_fp32_3.log {1: 909.46, 5: 909.68, 2: 908.85, 3: 908.63}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n8g/bert-base-adam-training_b32_fp32_4.log {1: 909.46, 5: 909.68, 2: 908.85, 3: 908.63, 4: 907.84}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n4g/bert-base-adam-training_b32_fp32_1.log {1: 457.03}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n4g/bert-base-adam-training_b32_fp32_5.log {1: 457.03, 5: 455.72}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n4g/bert-base-adam-training_b32_fp32_2.log {1: 457.03, 5: 455.72, 2: 455.7}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n4g/bert-base-adam-training_b32_fp32_3.log {1: 457.03, 5: 455.72, 2: 455.7, 3: 455.62}
/workspace/examples/bert/test_scripts/ngc_bert_b32/pytorch/1n4g/bert-base-adam-training_b32_fp32_4.log {1: 457.03, 5: 455.72, 2: 455.7, 3: 455.62, 4: 455.64}
{'bert-base-adam-training': {'1n1g': {'average_speed': 95.71,
                                      'batch_size_per_device': 32,
                                      'median_speed': 119.61,
                                      'speedup': 1.0},
                             '1n2g': {'average_speed': 221.24,
                                      'batch_size_per_device': 32,
                                      'median_speed': 221.18,
                                      'speedup': 1.85},
                             '1n4g': {'average_speed': 455.94,
                                      'batch_size_per_device': 32,
                                      'median_speed': 455.7,
                                      'speedup': 3.81},
                             '1n8g': {'average_speed': 908.89,
                                      'batch_size_per_device': 32,
                                      'median_speed': 908.85,
                                      'speedup': 7.6}}}
Saving result to ./result/_result.json
```

## 性能结果 Performance

该小节提供针对 NVIDIA PyTorch 框架的 BERT 模型测试的性能结果和完整 log 日志。

### FP32 & W/O XLA

- #### BERT-Base batch_size = 32

| gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
| ---------------- | --------------------- | ------------------ | ------- |
| 1                | 32                    | 119.61             | 1.00    |
| 2                | 32                    | 221.18             | 1.85    |
| 4                | 32                    | 455.7              | 3.81    |
| 8                | 32                    | 908.85             | 7.6     |

- #### BERT-Base batch_size = 48

| gpu_num_per_node | batch_size_per_device | samples/s(PyTorch) | speedup |
| ---------------- | --------------------- | ------------------ | ------- |
| 1                | 48                    | 122.91             | 1.00    |
| 2                | 48                    | 230.19             | 1.87    |
| 4                | 48                    | 469.75             | 3.82    |
| 8                | 48                    | 938.46             | 7.64    |

NVIDIA的 PyTorch 官方测评结果详见 [BERT For PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#training-performance-results)

详细 Log 信息可下载：[ngc_PyTorch_bert.zip]()
