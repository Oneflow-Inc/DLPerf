# 【DLPerf】DeepSpeed - BERT测评

# Overview

本次复现采用了微软[DeepSpeed官方仓库](https://github.com/microsoft/DeepSpeed/tree/1afca8f722fbcedf4b0ec0bf9e165d60564a7bba)中的[BERT-base](https://github.com/microsoft/DeepSpeedExamples/tree/ba63ad0fa861d28b3b33bc2c20f702647403e258/bing_bert)，目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试覆盖了FP32、FP16混合精度，后续将持续维护，增加更多方式的测评。



# Environment

## 系统

- 系统：Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- 显卡：Tesla V100-SXM2-16GB x 8
- 驱动：NVIDIA 440.33.01
- CUDA：10.2
- cuDNN：7.6.5
- NCCL：2.7.3

## 框架

- **torch 1.6.0 **

## Feature support matrix

| Feature                       | ResNet-50 v1.5 Paddle |
| ----------------------------- | --------------------- |
| Multi-node,multi-gpu training | Yes                   |
| NVIDIA NCCL                   | Yes                   |
| Mixed precision               | Yes                   |

# Quick Start

## 项目代码

- 微软[DeepSpeed官方仓库](https://github.com/microsoft/DeepSpeed/tree/1afca8f722fbcedf4b0ec0bf9e165d60564a7bba)
  - [BERT](https://github.com/microsoft/DeepSpeedExamples/tree/ba63ad0fa861d28b3b33bc2c20f702647403e258/bing_bert)

下载官方源码：

```shell
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed && git checkout 1afca8f722fbcedf4b0ec0bf9e165d60564a7bba
cd DeepSpeedExamples/bing_bert
```


将本页面中所有.json配置文件、scripts文件夹中的.sh脚本文件全部放入：`bing_bert/`路径下。



## 依赖安装

如果您是通过`docker pull deepspeed/deepspeed:latest`拉取了DeepSpeed官方镜像，则可通过下面命令启动容器：

```
docker pull deepspeed/deepspeed:latest
docker run -it -d  -p 12345:22 --shm-size=16g --ulimit memlock=-1 --privileged  \
--name deepspeed --cap-add=IPC_LOCK  \
-v /datasets/bert/deepspeed/data/test:/datasets  \
deepspeed/deepspeed:latest
```

并且容器内已安装好各种依赖，无需额外进行下面的安装。如果您是在物理机上，则需要进行下面的步骤来安装环境。

### 创建环境

```shell
# 创建conda环境
conda create -n deepspeed python=3.7.9
conda activate deepspeed
# 安装pytorch
python3 -m pip install torch==1.6.0 -i https://mirror.baidu.com/pypi/simple
python3 -m pip install torchvision==0.7.0 matplotlib==3.3.2
sudo apt install pdsh
```

修改`DeepSpeed/requirements/requirements.txt`，将前两行注释掉：

```
# torch>=1.2
# torchvision>=0.4.0
```

安装BERT训练相关依赖

```
# DeepSpeed主目录下执行
python3 -m pip install -r requirements/requirements.txt
python3 -m pip install -r requirements/requirements-dev.txt
```

### 安装deepspeed

conda deepspeed环境下执行：`bash install.sh` ，命令执行后会编译并安装一系列whl包如：apex,deepspeed等，过程中可能会报错：

![error.png](https://cdn.nlark.com/yuque/0/2020/png/216914/1602677523508-d71b79b8-625c-453a-9d64-c75d84afba79.png)

报错原因在于，脚本中执行的pip使用了本地.local中的pip，而其版本和conda环境deepspeed下的python3.7不同，导致报错：`ERROR: apex-0.1-cp37-cp37m-linux_x86_64.whl is not a supported wheel on this platform`

如报错，可修改脚本install.sh[第153行](https://github.com/microsoft/DeepSpeed/blob/1afca8f722fbcedf4b0ec0bf9e165d60564a7bba/install.sh#L153)，改为：

```
PIP_SUDO="python3  -m "
```

后重新执行`bash install.sh`

编译耗时大约几分钟，成功后：

![success.png](https://cdn.nlark.com/yuque/0/2020/png/216914/1602678788832-9819f36a-0b68-4a4c-a9dd-3b6b5a24fc03.png)



## NCCL

pytorch的分布式训练底层依赖NCCL库，需要从[NVIDIA-NCCL官网下载](https://developer.nvidia.com/nccl/nccl-download)并安装和操作系统、CUDA版本适配的NCCL。本次测试中安装2.7.3版本的NCCL：

```shell
sudo dpkg -i nccl-repo-ubuntu1604-2.7.3-ga-cuda10.2_1-1_amd64.deb
sudo apt update
sudo apt install libnccl2=2.7.3-1+cuda10.2 libnccl-dev=2.7.3-1+cuda10.2
```

## 数据集

### 训练集

本次训练使用Wikipedia数据集，并根据NVIDIA官方提供的脚本制作转换为.hdf5格式，详见：[NVIDIA-quick-start-guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide)。
### 词表文件

由于直接运行训练，程序会自动从s3.amazonaws.com下载词表文件(vocab.txt)，但速度很慢，故我们可以手动下载词表文件并放入新建文件夹`bing_bert/data`下，（直接运行训练，程序会自动从亚马逊amazonaws自动所有文件，但速度很慢）词表文件下载链接见：[tokenization.py](https://github.com/microsoft/DeepSpeedExamples/blob/ba63ad0fa861d28b3b33bc2c20f702647403e258/bing_bert/pytorch_pretrained_bert/tokenization.py)。下载完成并将词表文件存入`bing_bert/data`后，注释掉[tokenization.py Line:30]([tokenization.py](https://github.com/microsoft/DeepSpeedExamples/blob/ba63ad0fa861d28b3b33bc2c20f702647403e258/bing_bert/pytorch_pretrained_bert/tokenization.py#L30)) 的`PRETRAINED_VOCAB_ARCHIVE_MAP{}`，修改如下：

```python3
CACHE_DIR = "/your/path/to/DeepSpeed/DeepSpeedExamples/bing_bert/data/"
PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased':
    CACHE_DIR+"bert-base-uncased-vocab.txt",
    'bert-large-uncased':
    CACHE_DIR+"bert-large-uncased-vocab.txt",
    'bert-base-cased':
    CACHE_DIR+"bert-base-cased-vocab.txt",
    'bert-large-cased':
    CACHE_DIR+"bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased':
    CACHE_DIR+"bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased':
    CACHE_DIR+"bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese':
    CACHE_DIR+"bert-base-chinese-vocab.txt",
}
```

修改完成后，程序将从本地加载所需词表文件。


# Training

集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里默认设置batch size为32，分别在1机1卡～4机32卡的情况下进行了多组训练。

训练前，我们还需要做一些准备工作：

1.安装依赖：`python3 -m pip install boto3 h5py`（不安装直接跑bert pretraining会报错)

2.修改deepdpeed_train.py[Line 197](https://github.com/microsoft/DeepSpeedExamples/blob/ba63ad0fa861d28b3b33bc2c20f702647403e258/bing_bert/deepspeed_train.py#L197)如下：

```
else:
     epoch_step += 1
     # Call DeepSpeed engine step on micro steps
     model.network.step()
```

以使训练120iter后，程序会自动退出。

## 单机

`DeepSpeed/DeepSpeedExamples/bing_bert/`目录下，执行脚本：

```shell
bash run_single_node.sh
```

对单机1卡、4卡、8卡分别做5组测试，默认测试fp32精度，batch_size=32。

### 混合精度

可以通过参数指定fp16及batch_size：

```shell
bash run_single_node.sh 32 fp16
```

也可以自行指定精度以及batch_size：`bash run_single_node.sh 128 fp16`,`bash run_single_node.sh 64 fp32`。



## 2机16卡

2机、4机等多机情况下，需要在所有机器节点上相同路径准备同样的数据集、以完成分布式训练；此外，还需配置各个机器节点直接的ssh免密登录，配置完成后，我们在`deepSpeed/DeepSpeedExamples/bing_bert/`目录下新建hosts文件，例如deepspeed_hosts：

```
NODE1 slots=8
NODE1 slots=8
```

此文件，指定了2个节点NODE1和NODE2，每个节点使用8块GPU进行训练。

在主节点(NODE1节点)`deepSpeed/DeepSpeedExamples/bing_bert/`目录下执行脚本:

```shell
bash run_two_node.sh
```

即可运行2机16卡的训练，同样默认测试5组（fp32精度，batch_size=32）

### 混合精度

可以通过参数指定fp16及batch_size：

```shell
bash run_two_node.sh 128  fp16
```

## 4机32卡

流程同上，在主节点上执行：

```shell
bash run_multi_node.sh
```

以运行4机32卡的训练，默认测试5组（fp32精度，batch_size=32）。

### 混合精度

可以通过参数指定fp16及batch_size：

```shell
bash run_multi_node.sh 128 fp16
```

#### 注：FP32精度下的多机训练过程中会报错：

```shell
vs003: Traceback (most recent call last):
vs003:   File "deepspeed_train.py", line 540, in <module>
vs003:     main()
vs003:   File "deepspeed_train.py", line 533, in main
vs003:     run(args, model, optimizer, start_epoch)
vs003:   File "deepspeed_train.py", line 499, in run
vs003:     train(args, index, model, optimizer, pretrain_dataset_provider)
vs003:   File "deepspeed_train.py", line 187, in train
vs003:     report_step_metrics(args, lr_this_step, unscaled_loss,
vs003: UnboundLocalError: local variable 'lr_this_step' referenced before assignment
```

此报错为官方代码中尚未修复的bug，详见：

https://github.com/microsoft/DeepSpeedExamples/issues/53

https://github.com/microsoft/DeepSpeed/issues/426



# Result

## 吞吐率及加速比

执行以下命令，即可计算各种测试配置下的吞吐率及加速比：

```shell
python extract_deepspeed_logs_time.py  --log_dir=logs/deepspeed/bert/bz32 --batch_size_per_device=32
```

输出：

```shell
python extract_deepspeed_logs.py  --log_dir=logs/deepspeed/bert/bz32 --batch_size_per_device=32
logs/deepspeed/bert/bz32/4n8g/bert_b32_fp32_4.log {4: 4891.77}
logs/deepspeed/bert/bz32/4n8g/bert_b32_fp32_3.log {4: 4891.77, 3: 4903.74}
logs/deepspeed/bert/bz32/4n8g/bert_b32_fp32_5.log {4: 4891.77, 3: 4903.74, 5: 4900.76}
logs/deepspeed/bert/bz32/4n8g/bert_b32_fp32_2.log {4: 4891.77, 3: 4903.74, 5: 4900.76, 2: 4899.37}
logs/deepspeed/bert/bz32/4n8g/bert_b32_fp32_1.log {4: 4891.77, 3: 4903.74, 5: 4900.76, 2: 4899.37, 1: 4873.92}
logs/deepspeed/bert/bz32/1n8g/bert_b32_fp32_4.log {4: 1150.49}
logs/deepspeed/bert/bz32/1n8g/bert_b32_fp32_3.log {4: 1150.49, 3: 1186.46}
logs/deepspeed/bert/bz32/1n8g/bert_b32_fp32_5.log {4: 1150.49, 3: 1186.46, 5: 1146.9}
logs/deepspeed/bert/bz32/1n8g/bert_b32_fp32_2.log {4: 1150.49, 3: 1186.46, 5: 1146.9, 2: 1145.22}
logs/deepspeed/bert/bz32/1n8g/bert_b32_fp32_1.log {4: 1150.49, 3: 1186.46, 5: 1146.9, 2: 1145.22, 1: 1147.9}
logs/deepspeed/bert/bz32/1n4g/bert_b32_fp32_4.log {4: 587.55}
logs/deepspeed/bert/bz32/1n4g/bert_b32_fp32_3.log {4: 587.55, 3: 575.07}
logs/deepspeed/bert/bz32/1n4g/bert_b32_fp32_5.log {4: 587.55, 3: 575.07, 5: 572.11}
logs/deepspeed/bert/bz32/1n4g/bert_b32_fp32_2.log {4: 587.55, 3: 575.07, 5: 572.11, 2: 573.84}
logs/deepspeed/bert/bz32/1n4g/bert_b32_fp32_1.log {4: 587.55, 3: 575.07, 5: 572.11, 2: 573.84, 1: 577.14}
logs/deepspeed/bert/bz32/1n1g/bert_b32_fp32_4.log {4: 147.81}
logs/deepspeed/bert/bz32/1n1g/bert_b32_fp32_3.log {4: 147.81, 3: 147.93}
logs/deepspeed/bert/bz32/1n1g/bert_b32_fp32_5.log {4: 147.81, 3: 147.93, 5: 143.8}
logs/deepspeed/bert/bz32/1n1g/bert_b32_fp32_2.log {4: 147.81, 3: 147.93, 5: 143.8, 2: 143.61}
logs/deepspeed/bert/bz32/1n1g/bert_b32_fp32_1.log {4: 147.81, 3: 147.93, 5: 143.8, 2: 143.61, 1: 148.42}
logs/deepspeed/bert/bz32/2n8g/bert_b32_fp32_4.log {4: 2273.68}
logs/deepspeed/bert/bz32/2n8g/bert_b32_fp32_3.log {4: 2273.68, 3: 2267.55}
logs/deepspeed/bert/bz32/2n8g/bert_b32_fp32_5.log {4: 2273.68, 3: 2267.55, 5: 2368.65}
logs/deepspeed/bert/bz32/2n8g/bert_b32_fp32_2.log {4: 2273.68, 3: 2267.55, 5: 2368.65, 2: 2264.69}
logs/deepspeed/bert/bz32/2n8g/bert_b32_fp32_1.log {4: 2273.68, 3: 2267.55, 5: 2368.65, 2: 2264.69, 1: 2266.27}
{'bert': {'1n1g': {'average_speed': 146.31,
                   'batch_size_per_device': 32,
                   'median_speed': 147.81,
                   'speedup': 1.0},
          '1n4g': {'average_speed': 577.14,
                   'batch_size_per_device': 32,
                   'median_speed': 575.07,
                   'speedup': 3.89},
          '1n8g': {'average_speed': 1155.39,
                   'batch_size_per_device': 32,
                   'median_speed': 1147.9,
                   'speedup': 7.77},
          '2n8g': {'average_speed': 2288.17,
                   'batch_size_per_device': 32,
                   'median_speed': 2267.55,
                   'speedup': 15.34},
          '4n8g': {'average_speed': 4893.91,
                   'batch_size_per_device': 32,
                   'median_speed': 4899.37,
                   'speedup': 33.15}}}
Saving result to ./result/bz32_result.json
```

## 计算规则

### 1.测速脚本

- extract_paddle_logs.py
- extract_paddle_logs_time.py

两个脚本略有不同，得到的结果稍有误差：

extract_paddle_logs.py根据官方在log中打印的速度，在120个iter中，排除前20iter，取后100个iter的速度做平均；

extract_paddle_logs_time.py则根据log中打印出的时间，排除前20iter取后100个iter的实际运行时间计算速度。

README展示的是extract_paddle_logs.py的计算结果。

### 2.均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度

### 3.加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5

## BERT-Base  FP32

### batch size=32 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 147.81    | 1       |
| 1        | 4       | 575.07    | 3.89    |
| 1        | 8       | 1147.9    | 7.77    |
| 2        | 16      | 2267.55   | 15.34   |
| 4        | 32      | 4899.37   | 33.15   |

### batch size=64 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 152.32    | 1       |
| 1        | 4       | 601.64    | 3.95    |
| 1        | 8       | 1197.91   | 7.86    |
| 2        | 16      | 2318.82   | 15.22   |
| 4        | 32      | 4510.15   | 29.61   |



## BERT-Base  FP16

### batch size=64 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 565.3     | 1       |
| 1        | 4       | 2271.61   | 4.02    |
| 1        | 8       | 4512.68   | 7.98    |
| 2        | 16      | 8944.0    | 15.82   |
| 4        | 32      | 16401.67  | 29.01   |


### batch size=128 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 607.12    | 1       |
| 1        | 4       | 2412.1    | 3.97    |
| 1        | 8       | 4863.79   | 8.01    |
| 2        | 16      | 9892.88   | 16.29   |
| 4        | 32      | 16809.43  | 27.69   |

### batch size=160 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 619.73    | 1       |
| 1        | 4       | 2528.53   | 4.08    |
| 1        | 8       | 4953.73   | 7.99    |
| 2        | 16      | 10122.54  | 16.33   |
| 4        | 32      | 17751.63  | 28.64   |

## 完整日志

- [bert_fp32.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/DeepSpeed/bert/bert_fp32.zip)
- [bert_fp16.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/DeepSpeed/bert/bert_fp16.zip)
