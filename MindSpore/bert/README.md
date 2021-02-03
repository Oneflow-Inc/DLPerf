# 【DLPerf】MindSpore-BERT测评

# Overview

本次复现采用了[MindSpore官方仓库](https://gitee.com/mindspore/mindspore/tree/d9db5bf730ee7aa252eb7df41ffad09501acbe44)中的[BERT](https://gitee.com/mindspore/mindspore/tree/d9db5bf730ee7aa252eb7df41ffad09501acbe44/model_zoo/official/nlp/bert)，目的在于速度测评，同时根据测速结果给出1机、2机、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试已覆盖 FP32、FP16混合精度以及图算融合（Graph Kernel Fusion，类似 XLA 的图优化/算子融合技术，在本文档后续简称 GKF），后续将持续维护，增加更多方式的测评。


# Environment

## 系统

- 系统：Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- 显卡：Tesla V100-SXM2-16GB x 8
- 驱动：NVIDIA 440.33.01
- CUDA：10.2
- cuDNN：7.6.5

## 容器

- Ubuntu18.04
- Python 3.7
- CUDA 10.1.243
- OpenMPI 4.0.3

## 框架

- **MindSpore 1.1.0**

## Feature support matrix

| Feature            | BERT-Base MindSpore       |
| ------------------ | ------------------------- |
| Mpi Multi-gpu      | Yes                       |
| Mpi Multi-node     | Yes                       |
| Automatic mixed precision (AMP) | Yes                       |

# Quick Start

## 项目代码

- [MindSpore官方仓库](https://gitee.com/mindspore/mindspore/tree/d9db5bf730ee7aa252eb7df41ffad09501acbe44)
  - [BERT项目主页](https://gitee.com/mindspore/mindspore/tree/d9db5bf730ee7aa252eb7df41ffad09501acbe44/model_zoo/official/nlp/bert)

下载官方源码：

```shell
git clone https://gitee.com/mindspore/mindspore.git
cd mindspore/
git checkout r1.1
git reset d9db5bf730ee7aa252eb7df41ffad09501acbe44 --hard
cd model_zoo/official/nlp/bert/
```

1.将本页面scripts路径下的脚本：`run_single_node.sh`、`run_multi_node.sh`放入model_zoo/official/nlp/bert/路径下；

2.将本页面scripts路径下的其余脚本：`run_standalone_pretrain_for_gpu.sh`、`run_distributed_pretrain_for_gpu.sh`放入model_zoo/official/nlp/bert/scripts/下；

3.修改代码脚本，将本页面`run_pretrain.py`放入model_zoo/official/nlp/bert/路径下；

或者按如下说明手动修改官方原有`run_pretrain.py`：

将 model_zoo/official/nlp/bert/run_pretrain.py 138 行：
```shell
# line 138
    args_opt = parser.parse_args()
```
替换为：

```shell
    # line 138
    parser.add_argument("--optimizer", type=str, default="AdamWeightDecay", choices=["AdamWeightDecay", "Lamb", "Momentum"],
                        help="Optimizer, default is AdamWeightDecay.")
    parser.add_argument("--enable_global_norm", type=str, default="true", choices=["true", "false"],
                        help="Enable gloabl norm for grad clip, default is true.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"],
                        help="dtype, default is fp32.")

    args_opt = parser.parse_args()
    cfg.optimizer = args_opt.optimizer
    cfg.batch_size = args_opt.batch_size
    cfg.enable_global_norm = True if args_opt.enable_global_norm == "true" else False
    bert_net_cfg.compute_type = mstype.float32 if args_opt.dtype== "fp32" else mstype.float16
    logger.warning("\nargs_opt: {}".format(args_opt))
    logger.warning("\ncfg: {}".format(cfg))

```
增加输入参数。

将 model_zoo/official/nlp/bert/run_pretrain.py 179 行至 187 行：
```shell
    # line 179
    if args_opt.device_target == 'GPU' and bert_net_cfg.compute_type != mstype.float32 and \
    not is_auto_enable_graph_kernel:
    warning_message = 'Gpu only support fp32 temporarily, run with fp32.'
    bert_net_cfg.compute_type = mstype.float32
    if args_opt.enable_lossscale == "true":
        args_opt.enable_lossscale = "false"
        warning_message = 'Gpu only support fp32 temporarily, run with fp32 and disable lossscale.'
    logger.warning(warning_message)

```
删除，避免参数间互相影响。

## 容器

本次测评采用的是MindSpore官方提供的Docker镜像，您可以
参考[MindSpore官方文档](https://gitee.com/mindspore/mindspore/tree/d9db5bf730ee7aa252eb7df41ffad09501acbe44/#docker%E9%95%9C%E5%83%8F)GPU部分
**获取项目镜像**

对于`GPU`后端，请确保`nvidia-container-toolkit`已经提前安装，以下是`Ubuntu`用户安装指南：

```bash
DISTRIBUTION=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$DISTRIBUTION/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
sudo systemctl restart docker
```

编辑文件 daemon.json:

```bash
$ vim /etc/docker/daemon.json
{   
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }   
    }   
}   
```

再次重启docker:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

使用以下命令获取镜像：

```bash
docker pull mindspore/mindspore-gpu:1.1.0
```

根据项目镜像**启动容器**
```shell
docker run -it \
    --runtime=nvidia \
    --privileged=true \
    --net=host \
    --cap-add=IPC_LOCK \
    --device=/dev/infiniband \
    --name mindspore_bert \
    -v /dev/shm:/dev/shm \
    -v $PWD:/workspace/bert \
    -v /home/leinao/dataset/wiki:/workspace/bert/data/wiki \
    -v $PWD/results:/results \
    mindspore/mindspore-gpu:1.1.0 /bin/bash
```

## 数据集

数据集采用TFRecord格式的[Wikipedia](https://dumps.wikimedia.org/)，制作方式请参考：[MindSpore官方仓库说明](https://gitee.com/mindspore/mindspore/tree/d9db5bf730ee7aa252eb7df41ffad09501acbe44/model_zoo/official/nlp/bert#dataset) ；

数据集具体制作步骤及脚本见：[TensorFlow 2.x-BERT测评](https://github.com/Oneflow-Inc/DLPerf/tree/master/TensorFlow/bert#%E6%95%B0%E6%8D%AE%E9%9B%86)

## SSH配置(可选)

单机情况下无需配置ssh服务，需要测试2机、4机等情况下时，则需要安装docker容器间的ssh服务，配置ssh免密登录，保证分布式horovod/mpi脚本运行时可以在多机间互联。

配置过程详见文档[SSH配置](https://github.com/Oneflow-Inc/DLPerf/tree/master/NVIDIADeepLearningExamples/TensorFlow/LanguageModeling/BERT#ssh%E9%85%8D%E7%BD%AE%E5%8F%AF%E9%80%89)。

## IB驱动安装（可选）

如果服务器之间支持IB(**InfiniBand**)网络，则可以安装IB驱动，使得多机情况下各个节点间的通信速率明显提升，从而加速框架在多机环境下的训练，提升加速比。

配置过程详见文档[IB驱动安装](https://github.com/Oneflow-Inc/DLPerf/tree/dev_mindspore/NVIDIADeepLearningExamples/TensorFlow/LanguageModeling/BERT#ib%E9%A9%B1%E5%8A%A8%E5%AE%89%E8%A3%85%E5%8F%AF%E9%80%89)。


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
docker exec -it mindspore_bert /bin/bash
cd /workspace/bert
bash run_single_node.sh 32 fp32 5
```

执行脚本测试fp32+batch size32，对单机1卡、2卡、4卡、8卡分别做5次测试，也可以指定其他batch_size参数进行测试。

### 混合精度

指定dtype运行参数为fp16，以进行fp16混合精度测试，例如：

- batch size=64 使用fp16混合精度：

```shell
bash   run_single_node.sh 64 fp16 5
```

## 多机

2机、4机等多机情况下，需要在所有机器节点上相同路径准备同样的数据集、以完成分布式训练。由于配置了ssh免密，您只需要在一个节点上运行脚本即可执行多机训练。

如2机：NODE1='10.11.0.2'   NODE2='10.11.0.3' 的训练，需在两台机器上分别准备好数据集后，NODE1节点进入容器/workspace/bert下，执行脚本:

```shell
bash   run_multi_node.sh 32 fp32 5 2
```
即可运行2机16卡的训练，同样测试5次。

### 混合精度

指定dtype运行参数为fp16，以进行fp16混合精度测试，例如：

- batch size=64 使用fp16混合精度：

```shell
bash   run_multi_node.sh 64 fp16 5 2
```

## Result

### 吞吐率及加速比

执行以下命令，即可计算各种测试配置下的吞吐率及加速比：

```shell
python extract_mindspore_logs_time.py --log_dir=logs_fp32/mindspore/bert/bz32
```

输出：

```shell
logs_fp32/mindspore/bert/bz32/1n2g/bert_b32_fp32_5.log {5: 174.38}
logs_fp32/mindspore/bert/bz32/1n2g/bert_b32_fp32_1.log {5: 174.38, 1: 174.45}
logs_fp32/mindspore/bert/bz32/1n2g/bert_b32_fp32_2.log {5: 174.38, 1: 174.45, 2: 174.4}
logs_fp32/mindspore/bert/bz32/1n2g/bert_b32_fp32_4.log {5: 174.38, 1: 174.45, 2: 174.4, 4: 174.56}
logs_fp32/mindspore/bert/bz32/1n2g/bert_b32_fp32_3.log {5: 174.38, 1: 174.45, 2: 174.4, 4: 174.56, 3: 174.52}
logs_fp32/mindspore/bert/bz32/2n8g/bert_b32_fp32_5.log {5: 1309.42}
logs_fp32/mindspore/bert/bz32/2n8g/bert_b32_fp32_1.log {5: 1309.42, 1: 1309.64}
logs_fp32/mindspore/bert/bz32/2n8g/bert_b32_fp32_2.log {5: 1309.42, 1: 1309.64, 2: 1309.94}
logs_fp32/mindspore/bert/bz32/2n8g/bert_b32_fp32_4.log {5: 1309.42, 1: 1309.64, 2: 1309.94, 4: 1309.76}
logs_fp32/mindspore/bert/bz32/2n8g/bert_b32_fp32_3.log {5: 1309.42, 1: 1309.64, 2: 1309.94, 4: 1309.76, 3: 1309.7}
logs_fp32/mindspore/bert/bz32/1n1g/bert_b32_fp32_5.log {5: 112.22}
logs_fp32/mindspore/bert/bz32/1n1g/bert_b32_fp32_1.log {5: 112.22, 1: 112.18}
logs_fp32/mindspore/bert/bz32/1n1g/bert_b32_fp32_2.log {5: 112.22, 1: 112.18, 2: 112.12}
logs_fp32/mindspore/bert/bz32/1n1g/bert_b32_fp32_4.log {5: 112.22, 1: 112.18, 2: 112.12, 4: 112.18}
logs_fp32/mindspore/bert/bz32/1n1g/bert_b32_fp32_3.log {5: 112.22, 1: 112.18, 2: 112.12, 4: 112.18, 3: 112.21}
logs_fp32/mindspore/bert/bz32/4n8g/bert_b32_fp32_5.log {5: 2451.44}
logs_fp32/mindspore/bert/bz32/4n8g/bert_b32_fp32_1.log {5: 2451.44, 1: 2458.83}
logs_fp32/mindspore/bert/bz32/4n8g/bert_b32_fp32_2.log {5: 2451.44, 1: 2458.83, 2: 2450.83}
logs_fp32/mindspore/bert/bz32/4n8g/bert_b32_fp32_4.log {5: 2451.44, 1: 2458.83, 2: 2450.83, 4: 2458.03}
logs_fp32/mindspore/bert/bz32/4n8g/bert_b32_fp32_3.log {5: 2451.44, 1: 2458.83, 2: 2450.83, 4: 2458.03, 3: 2457.8}
logs_fp32/mindspore/bert/bz32/1n4g/bert_b32_fp32_5.log {5: 352.54}
logs_fp32/mindspore/bert/bz32/1n4g/bert_b32_fp32_1.log {5: 352.54, 1: 352.38}
logs_fp32/mindspore/bert/bz32/1n4g/bert_b32_fp32_2.log {5: 352.54, 1: 352.38, 2: 352.76}
logs_fp32/mindspore/bert/bz32/1n4g/bert_b32_fp32_4.log {5: 352.54, 1: 352.38, 2: 352.76, 4: 352.69}
logs_fp32/mindspore/bert/bz32/1n4g/bert_b32_fp32_3.log {5: 352.54, 1: 352.38, 2: 352.76, 4: 352.69, 3: 352.93}
logs_fp32/mindspore/bert/bz32/1n8g/bert_b32_fp32_5.log {5: 704.3}
logs_fp32/mindspore/bert/bz32/1n8g/bert_b32_fp32_1.log {5: 704.3, 1: 704.85}
logs_fp32/mindspore/bert/bz32/1n8g/bert_b32_fp32_2.log {5: 704.3, 1: 704.85, 2: 704.81}
logs_fp32/mindspore/bert/bz32/1n8g/bert_b32_fp32_4.log {5: 704.3, 1: 704.85, 2: 704.81, 4: 704.37}
logs_fp32/mindspore/bert/bz32/1n8g/bert_b32_fp32_3.log {5: 704.3, 1: 704.85, 2: 704.81, 4: 704.37, 3: 705.25}
{'bert': {'1n1g': {'average_speed': 112.18,
                   'batch_size_per_device': 32,
                   'median_speed': 112.18,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 174.46,
                   'batch_size_per_device': 32,
                   'median_speed': 174.45,
                   'speedup': 1.56},
          '1n4g': {'average_speed': 352.66,
                   'batch_size_per_device': 32,
                   'median_speed': 352.69,
                   'speedup': 3.14},
          '1n8g': {'average_speed': 704.72,
                   'batch_size_per_device': 32,
                   'median_speed': 704.81,
                   'speedup': 6.28},
          '2n8g': {'average_speed': 1309.69,
                   'batch_size_per_device': 32,
                   'median_speed': 1309.7,
                   'speedup': 11.67},
          '4n8g': {'average_speed': 2455.39,
                   'batch_size_per_device': 32,
                   'median_speed': 2457.8,
                   'speedup': 21.91}}}
Saving result to ./result/_result.json
```


### 计算规则

#### 1.测速脚本

- extract_mindspore_logs_time.py

extract_mindspore_logs_time.py根据log中打印出的耗时，排除前20iter取后100个iter来计算速度。

#### 2.均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度。

#### 3.加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5

### BERT-Base  FP32

#### batch size=32 & without Graph Kernel Fusion

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 112.18    | 1       |
| 1        | 2       | 174.45    | 1.56    |
| 1        | 4       | 352.69    | 3.14    |
| 1        | 8       | 704.81    | 6.28    |
| 2        | 16      | 1309.7    | 11.67   |
| 4        | 32      | 2457.8    | 21.91   |

#### batch size=32 & with Graph Kernel Fusion

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 148.74    | 1       |
| 1        | 2       | 258.38    | 1.74    |
| 1        | 4       | 548.7     | 3.69    |
| 1        | 8       | 1106.81   | 7.44    |
| 2        | 16      | 1707.44   | 11.48   |
| 4        | 32      | 3423.44   | 23.02   |

#### batch size=64 & without Graph Kernel Fusion

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 117.99    | 1       |
| 1        | 2       | 205.54    | 1.74    |
| 1        | 4       | 413.71    | 3.51    |
| 1        | 8       | 827.83    | 7.02    |
| 2        | 16      | 1582.82   | 13.41   |
| 4        | 32      | 3051.3    | 25.86   |

#### batch size=64 & with Graph Kernel Fusion

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 153.51    | 1       |
| 1        | 2       | 284.69    | 1.85    |
| 1        | 4       | 588.12    | 3.83    |
| 1        | 8       | 1181.56   | 7.7     |
| 2        | 16      | 2044.75   | 13.32   |
| 4        | 32      | 4080.97   | 26.58   |

#### batch size=96 & with Graph Kernel Fusion

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 154.68    | 1       |
| 1        | 2       | 294.01    | 1.9     |
| 1        | 4       | 601.62    | 3.89    |
| 1        | 8       | 1207.47   | 7.81    |
| 2        | 16      | 2185.27   | 14.13   |
| 4        | 32      | 4350.6    | 28.13   |

注：
- 以32为最小单位，关闭Graph Kernel Fusion时，最大batch size为64，否则会OOM(out of memory)。
- 以32为最小单位，打开Graph Kernel Fusion时，最大batch size为96，否则会OOM(out of memory)。

### BERT-Base  FP16

#### batch size=64 & without Graph Kernel Fusion

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 185.78    | 1       |
| 1        | 2       | 347.73    | 1.87    |
| 1        | 4       | 698.06    | 3.76    |
| 1        | 8       | 1376.28   | 7.41    |
| 2        | 16      | 2525.83   | 13.6    |
| 4        | 32      | 4659.76   | 25.08   |

#### batch size=128 & without Graph Kernel Fusion

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 216.49    | 1       |
| 1        | 2       | 417.56    | 1.93    |
| 1        | 4       | 838.24    | 3.87    |
| 1        | 8       | 1652.89   | 7.63    |
| 2        | 16      | 3171.2    | 14.65   |
| 4        | 32      | 6068.55   | 28.03   |

注：
- 以32为最小单位，关闭Graph Kernel Fusion时，最大batch size为128，否则会OOM(out of memory)。
- 本次测评使用的MindSpore官方代码在使用FP16混合精度训练Bert时，若同时打开dynamic loss scale和Graph Kernel Fusion，会返回“Build akg kernel op failed”错误，因此暂时没有该配置的结果。

### 完整日志

- [bert_fp32.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MindSpore/bert/bert_fp32.zip) 
- [bert_fp16.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MindSpore/bert/bert_fp16.zip) 
- [bert_fp32_gkf.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MindSpore/bert/bert_fp32_gkf.zip) 
