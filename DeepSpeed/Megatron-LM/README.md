# DeepSpeed-Megatron-LM测评

## 概述 Overview

本测评基于微软[DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/a79272cc8b8f0c5b66c803e581a1355341eacb77) 仓库中的[Megatron-LM](https://github.com/microsoft/DeepSpeedExamples/tree/a79272cc8b8f0c5b66c803e581a1355341eacb77/Megatron-LM)实现，框架依赖[DeepSpeed](https://github.com/microsoft/DeepSpeed/tree/7d4d742bf03f8e1707130391e0b39bd6d93a702a) 以及pytorch，基于以上环境，对gpt-2 small在单机单卡～4机32卡情况下进行了多组测试。测评主要目标在于评价ZeRO不同优化阶段时的训练速度、加速比情况。

测评背景：DeepSpeed是一个深度学习优化库，使分布式训练简单、高效和有效，DeepSpeed实现了[论文](https://arxiv.org/abs/1910.02054)中提出的ZeRo内存优化技术，减少了显存占用以支持更大模型。

## 环境 Environment

所有的测试都是在4台配置了8张 V100-SXM2-16GB GPU的服务器中，主要硬软件配置信息如下：

- #### 硬件

  - GPU：8x Tesla V100-SXM2-16GB
  - InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
  - Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
  - Memory 384G

- #### 软件

  - 驱动：Driver Version: 460.67
  
  - 系统：[ Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)](http://releases.ubuntu.com/16.04/)
  
  - CUDA：10.2
  
  - cuDNN：7.6.5
  
  - NCCL：2.7.3
  
  - Python：3.7.9
  
- #### 框架

  - **pytorch 1.6.0** 
  - **deepspeed 0.3.0+7d4d742**  

- #### GPU拓扑

```python

		GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  CPU Affinity
GPU0     X      NV1     NV1     NV2     NV2     SYS     SYS     SYS     NODE    0-11,24-35
GPU1    NV1      X      NV2     NV1     SYS     NV2     SYS     SYS     NODE    0-11,24-35
GPU2    NV1     NV2      X      NV2     SYS     SYS     NV1     SYS     PIX     0-11,24-35
GPU3    NV2     NV1     NV2      X      SYS     SYS     SYS     NV1     PIX     0-11,24-35
GPU4    NV2     SYS     SYS     SYS      X      NV1     NV1     NV2     SYS     12-23,36-47
GPU5    SYS     NV2     SYS     SYS     NV1      X      NV2     NV1     SYS     12-23,36-47
GPU6    SYS     SYS     NV1     SYS     NV1     NV2      X      NV2     SYS     12-23,36-47
GPU7    SYS     SYS     SYS     NV1     NV2     NV1     NV2      X      SYS     12-23,36-47
mlx5_0  NODE    NODE    PIX     PIX     SYS     SYS     SYS     SYS      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```



## 快速开始 Quick Start

### 1.环境准备

#### 新建conda环境

推荐新建一个deepspeed的conda环境

```shell
# 创建conda虚拟环境
conda create -n deepspeed python=3.7.9
# 激活环境
conda activate deepspeed
# 设置镜像源
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 安装pytorch
python3 -m pip install torch==1.6.0
```

#### 源码下载

```shell
# 下载DeepSpeed仓库
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
git checkout 7d4d742bf03f8e1707130391e0b39bd6d93a702a
# 初始化DeepSpeedExamples仓库
git submodule update --init --recursive
```

下载完仓库源码后，需要编译安装deepspeed

```shell
bash install.sh
```

编译成功后，主要的pip库版本如下：

- deepspeed                     0.3.0+7d4d742
- torch                                1.6.0

### 2. 数据集准备

数据集使用Wikipedia数据集，并使用[官方ReadMe](https://github.com/microsoft/DeepSpeedExamples/tree/a79272cc8b8f0c5b66c803e581a1355341eacb77/Megatron-LM#data-sets)中推荐的方式进行了处理。

#### 下载wiki数据集

`wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2`

#### 数据集解析

使用[wikiextractor](https://github.com/attardi/wikiextractor)将原始文件解析成txt格式的文本文件。

```python
python -m wikiextractor.WikiExtractor /your_path_to/enwiki-latest-pages-articles.xml.bz2  --json --no_templates -o /datasets/wiki/enwiki --processes 48
```

#### 划分句子并存为json格式

划分原始数据集中的句子，并存为相应的json格式文件。

DeepSpeed/DeepSpeedExamples/Megatron-LM目录下修改presplit_sentences_json.py中的代码，以支持遍历输入wiki数据集文件夹中的所有文件，并将其依次划分并存为相应.json格式到指定的输出目录。

`vim scripts/presplit_sentences_json.py`

将代码改为如下所示：

```python
import sys
import json
import nltk
import os

nltk.download('punkt')

line_seperator = "\n"
def write_to_json(input_file, output_file):
  with open(input_file, 'r') as ifile:
    with open(output_file, "w") as ofile:
      for doc in ifile.readlines():
        parsed = json.loads(doc)
        sent_list = []
      for line in parsed['text'].split('\n'):
          if line != '\n':
              sent_list.extend(nltk.tokenize.sent_tokenize(line))
      parsed['text'] = line_seperator.join(sent_list)
      ofile.write(json.dumps(parsed)+'\n')

"""
Usage:
python3 scripts/presplit_sentences_json.py  <input_dir>  <output_dir>

Such as:
python3 scripts/presplit_sentences_json.py /datasets/wiki/enwiki  /datasets/wiki/gpt2/json
"""
txtpath = sys.argv[1] #  "/datasets/wiki/enwiki"
jsonpath = sys.argv[2] #  "/datasets/wiki/gpt2/json"
assert os.path.exists(txtpath) and os.path.exists(jsonpath), "Make sure input and output dir exists!"
dirlist = os.listdir(txtpath)
for i in range(0, len(dirlist)):
  dpath = os.path.join(txtpath,dirlist[i])
  files = os.listdir(dpath)
  if len(files)>0:
    outdir = os.path.join(jsonpath,dirlist[i])
    os.makedirs(outdir, exist_ok=True) 
    print("dirpath:", dpath)
    for j in range(0, len(files)):
      fpath = os.path.join(dpath,files[j])
      print("file:", fpath)
      write_to_json(fpath, outdir+"/" + files[j] + ".json")
```

运行：

`python3 scripts/presplit_sentences_json.py /datasets/wiki/enwiki /datasets/wiki/gpt2/json`

#### 划分数据集

处理好的json格式完整数据集，需要划分为训练集、测试集、验证集等。

```shell
cd DeepSpeed/DeepSpeedExamples/Megatron-LM
python3 scripts/split_json.py --input_files /datasets/wiki/gpt2/json/*/*.json --output_dir /datasets/wiki/gpt2/
```

#### 修改数据集路径

最后，需修改corpora.py中默认的PATH路径，将路径改为上一步划分数据集后train.json所在路径（此路径会在gpt2训练时用到）。

`cd DeepSpeed/DeepSpeedExamples/Megatron-LM`

`vim data_utils/corpora.py` 修改PATH为你本地train.json所在路径

```python
import os

class wikipedia(json_dataset):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    PATH = '/datasets/wiki/gpt2/train.json'
    assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"
    def __init__(self, **kwargs):
```



### 3. 脚本&配置

将本仓库scripts目录下的文件放入`DeepSpeed/DeepSpeedExamples/Megatron-LM/scripts`

- run_xxx_node.sh为单机～多机测试的启动脚本
- runner.sh为训练主脚本（会在启动脚本中被调用）
- ds_zero2_config.json为测试脚本相关的配置文件

如需要进行多机测试，需要在`DeepSpeed/DeepSpeedExamples/Megatron-LM`下新建集群的hosts文件，可参考本仓库中的deepspeed_hosts：

```shell
vs002 slots=8
vs003 slots=8
```

表示集群使用2台机器，每个机器使用8张GPU设备；4机及以上，相应地在deepspeed_hosts中增加集群配置即可。

### 4.测试

本次测试集群中有 4 台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有 8 张 V100 显卡， 每张显卡显存 16 GB。

#### 参数及配置

测试主脚本为 `runner.sh` ，其中，默认使用gpt-2 small的网络配置，主要参数说明如下：

- `BATCH_SIZE_PER_DEVICE` 为单卡batch_size，默认为8
- `NUM_WORKERS` 分布式训练集群中，机器节点数（单机情况下设为1；4机情况下设为4，根据情况设置）
- `NUM_GPUS_PER_WORKER`  每台机器使用的gpu数，默认为8
- `ZERO_STAGE` zero优化阶段，默认为0，可选0,1,2（3目前DeepSpeed框架暂不支持）
- `CHECKPOINT_ACTIVATIONS`  是否开启activation/gradient checkpointing优化，默认为off关闭
- `MP_SIZE` 模型并行度，可以在1～NUM_GPUS_PER_WORKER数字中设置（默认1为不开启模型并行）
- `ITER_NUM` 测试迭代的iter数，默认迭代200 iter

配置相关的参数如下：

```shell
MODEL=${1:-gpt2-small}
BATCH_SIZE_PER_DEVICE=${2:-8}
NUM_WORKERS=${3:-1}
NUM_GPUS_PER_WORKER=${4:-8}
ZERO_STAGE=${5:-2}
CHECKPOINT_ACTIVATIONS=${6:-"on"}
DTYPE=${7:-'fp16'}
TEST_NUM=${8:-1}
ITER_NUM=${9:-200
```

#### 单机测试

运行 `bach scripts/run_single_node.sh`  即可，默认测试条件为：gpt-2-small网络、batch size为8、fp16混合精度、zero-stage-2优化阶段，也可自定义参数，如不同的batch size和不同zero优化阶段： `bach scripts/run_single_node.sh gpt2-small 16 1`  。

#### 多机测试

多机测试时需要保证多机上的数据集及路径、ds_zero2_config.json配置完全一样，然后运行相应脚本即可，如2机，可运行： `bach scripts/run_two_node.sh`  ，4机可运行： `bach scripts/run_multi_node.sh`  。

#### 其他测试

除了以上测试外，还可以通过设置不同参数进行多种类型测试。如，可以将CHECKPOINT_ACTIVATIONS设置为off来测试关闭checkpointing的情况，由于checkpointing关闭后，内存占用较大，故可以相应地降低batch size（如设置为4）。

off-checkpointing 的测试脚本示例如下：

```shell
# 单机1卡
bash scripts/runner.sh  gpt2-small  4  1  1  0  off 
# 单机4卡
bash scripts/runner.sh  gpt2-small  4  1  4  0  off 
# 单机8卡
bash scripts/runner.sh  gpt2-small  4  1  8  0  off 
# 2机16卡
bash scripts/runner.sh  gpt2-small  4  2  8  0  off
# 4机32卡
bash scripts/runner.sh  gpt2-small  4  4  8  0  off
```

### 4. 吞吐率及加速比

执行以下命令，即可根据logs文件计算出训练吞吐率及加速比：

`python3 extract_deepspeed_logs.py --log_dir=./logs/deepspeed/gpt2-small/bz8`

输出：

```python3

```



## 测试结果 Result

### Gpt2-small & AMP

#### zero-stage-0

| node_num | batch_size_per_device | gpu_num_per_node | samples/s | speedup |
| -------- | --------------------- | ---------------- | --------- | ------- |
| 1        | 8                     | 1                | 22.65     | 1       |
| 1        | 8                     | 4                | 89.1      | 3.93    |
| 1        | 8                     | 8                | 178.09    | 7.86    |
| 2        | 8                     | 8                | 292.63    | 12.92   |
| 4        | 8                     | 8                |           |         |
| 1        | 16(max)               | 1                | 23.96     | 1       |
| 1        | 16                    | 4                | 95.16     | 3.97    |
| 1        | 16                    | 8                | 190.36    | 7.94    |
| 2        | 16                    | 8                | 332.98    | 13.9    |
| 4        | 16                    | 8                |           |         |

#### zero-stage-1

| node_num | batch_size_per_device | gpu_num_per_node | samples/s | speedup |
| -------- | --------------------- | ---------------- | --------- | ------- |
| 1        | 8                     | 1                | 21.93     | 1       |
| 1        | 8                     | 4                | 88.29     | 4.03    |
| 1        | 8                     | 8                | 176.66    | 8.06    |
| 2        | 8                     | 8                | 290.72    | 13.26   |
| 4        | 8                     | 8                |           |         |
| 1        | 16(max)               | 1                | 23.5      | 1       |
| 1        | 16                    | 4                | 94.96     | 4.04    |
| 1        | 16                    | 8                | 189.44    | 8.06    |
| 2        | 16                    | 8                | 334.97    | 14.25   |
| 4        | 16                    | 8                |           |         |

#### zero-stage-2

| node_num | batch_size_per_device | gpu_num_per_node | samples/s | speedup |
| -------- | --------------------- | ---------------- | --------- | ------- |
| 1        | 8                     | 1                | 22.08     | 1       |
| 1        | 8                     | 4                | 90.92     | 4.12    |
| 1        | 8                     | 8                | 183.7     | 8.32    |
| 2        | 8                     | 8                | 278.19    | 12.6    |
| 4        | 8                     | 8                |           |         |
| 1        | 16(max)               | 1                | 23.65     | 1       |
| 1        | 16                    | 4                | 96.07     | 4.06    |
| 1        | 16                    | 8                | 193.01    | 8.16    |
| 2        | 16                    | 8                | 330.08    | 13.96   |
| 4        | 16                    | 8                |           |         |



#### off-checkpointing测试结果

| date     | test_num | test_desc    | xn_xg_xdp_xmp_xbs | gpu_mem(mB)   | gpu_util(%) | throuthput(sample/sec) |
| -------- | -------- | ------------ | ----------------- | ------------- | ----------- | ---------------------- |
| 20201205 | test-1-3 | zero-stage-0 | 1n_8g_dp_4bs      | 15387         | 92          | 198.6                  |
|          | test-1-4 | zero-stage-1 | 1n_8g_dp_4bs      | 14199(↓7.7%)  | 87          | 195.8 (↓1.4%)          |

| date     | test_num  | test_desc    | xn_xg_xdp_xmp_xbs | gpu_mem(mB)        | gpu_util(%) | throuthput(sample/sec) |
| -------- | --------- | ------------ | ----------------- | ------------------ | ----------- | ---------------------- |
| 20201208 | test-13-1 | zero-stage-0 | 2n_8g_dp_4bs      | 15336              | 94          | 269                    |
|          | test-13-2 | zero-stage-1 | 2n_8g_dp_4bs      | 14005(-1331,8.7↓%) | 93          | 261(3.0↓%)             |

| date     | test_num  | test_desc        | xn_xg_xdp_xmp_xbs | gpu_mem(mB)        | gpu_util(%) | throuthput(sample/sec) |
| -------- | --------- | ---------------- | ----------------- | ------------------ | ----------- | ---------------------- |
| 20201209 | test-18-1 | zero-stage-0     | 4n_8g_dp_4bs      | 15336              | 95          | 533                    |
|          | test-18-2 | zero-stage-1     | 4n_8g_dp_4bs      | 13975(-1361,↓8.9%) | 92          | 506(↓5.3%)             |

注：xn_xg_xdp_xmp_xbs表示x node, x gpu, x data parallel, x model parallel, x batch size per gpu



### 日志下载

详细 Log 信息可点击下载：

- [deepspeed-off-checkpointing-logs.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/DeepSpeed/gpt2/deepspeed-off-checkpointing-logs.zip)




