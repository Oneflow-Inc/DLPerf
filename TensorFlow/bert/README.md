# 【DLPerf】TensorFlow 2.x-BERT测评

# Overview
本次复现采用了[Tensorflow官方仓库](https://github.com/tensorflow/models/tree/r2.3.0)中的tf2.x版[BERT](https://github.com/tensorflow/models/tree/r2.3.0/official/nlp/bert)的实现，复现的目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试已覆盖 FP32、FP16混合精度以及XLA，后续将持续维护，增加更多方式的测评。



# Environment
## 系统

- 系统：Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- 显卡：Tesla V100-SXM2-16GB x 8
- 驱动：NVIDIA 440.33.01
- CUDA：10.2
- cuDNN：7.6.5
- NCCL：2.7.3
## 框架

- **TensorFlow 2.3.0** 



# Quick Start
## 项目代码

- [TensorFlow官方仓库](https://github.com/tensorflow/models/tree/r2.3.0)
   - [BERT项目主页](https://github.com/tensorflow/models/tree/r2.3.0/official/nlp/bert)

下载官方源码：
```shell
git clone https://github.com/tensorflow/models.git
cd models/ && git checkout r2.3.0
cd official/nlp/bert
```

将本页面scripts文件夹中的脚本放入models/official/nlp/bert目录下。

修改[run_pretraining.py](https://github.com/tensorflow/models/blob/r2.3.0/official/nlp/bert/run_pretraining.py)以支持xla和多机测试，将以下部分：

```shell
# LINE 186
  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)
```
替换为：
```shell
# LINE 186
  from official.utils.misc import keras_utils
  keras_utils.set_session_config(enable_xla=FLAGS.enable_xla)

  distribution_utils.configure_cluster(
      FLAGS.worker_hosts,
      FLAGS.task_index)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      all_reduce_alg=FLAGS.all_reduce_alg,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)
```


## 框架安装
```shell
python -m pip install tensorflow==2.3.0 -i https://mirror.baidu.com/pypi/simple
```
若出现依赖包未安装错误，可使用pip安装对应包（参考models/official/requirements.txt，其中gin即gin-config）
## NCCL
TensorFlow的分布式训练底层依赖NCCL库，需要从[NVIDIA-NCCL官网下载](https://developer.nvidia.com/nccl/nccl-download)并安装和操作系统、CUDA版本适配的NCCL。

本次测试中安装2.7.3版本的NCCL：

```shell
sudo dpkg -i nccl-repo-ubuntu1604-2.7.3-ga-cuda10.2_1-1_amd64.deb
sudo apt update
sudo apt install libnccl2=2.7.3-1+cuda10.2 libnccl-dev=2.7.3-1+cuda10.2
```
## 数据集
BERT-Pretraining数据集采用Wikipedia制作，具体制作过程参考tensorflow官方ReadMe文档：[pre-training](https://github.com/tensorflow/models/tree/r2.3.0/official/nlp/bert#pre-training ) ，制作时使用了官方提供的代码：[create_pretraining_data.py](https://github.com/tensorflow/models/blob/r2.3.0/official/nlp/data/create_pretraining_data.py)

设置create_pretraining_data.py中的相应参数，其中LINE 34、LINE 38、LINE 41为必填参数：

```shell
# LINE 34
flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")
# LINE 38
flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")
# LINE 41
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
```

### 参数说明

- input_file 是原始txt文件，可以是wiki或者其他数据集的txt，可以是单个文件，可以是多个文件。

示例：

```shell
 '/datasets/bert/AA/wiki_00'
 '/datasets/bert/AA/wiki_00,/datasets/bert/AA/wiki_01'
'/datasets/bert/AA/wiki_00,/datasets/bert/AA/wiki_*'
```

- output_file是制作完成的tfrecord文件，同样可以是单个/多个，如：wiki_AA.tfrecord

- vocab_file是词表文件，如：uncased_L-12_H-768_A-12/vocab.txt

### 制作tfrecord数据集

准备好原始txt文件并设置相应参数后，执行以下脚本即可生成tfrecord数据集：

```shell
export PYTHONPATH=$PYTHONPATH:/your/path/to/models
cd /your/path/to/models/official/nlp/bert
python3  ../data/create_pretraining_data.py
```



# Training

集群中有4台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里设置batch size为32、48和64，在1机1卡～1机8卡的情况下进行了多组训练。


## 单机

修改环境变量PYTHONPATH:`export PYTHONPATH=$PYTHONPATH:/your/path/to/models`

`models/official/nlp/bert`目录下，设置`single_node_train.sh`脚本中的训练/配置参数，然后执行脚本:

```shell
bash run_single_node.sh
```
对单机1卡、2卡、4卡、8卡分别做5组测试。单机脚本默认的batch size为32，可以通过参数指定，如指定batch size为48或64：`bash run_single_node.sh 48`，`bash run_single_node.sh 64`


## 2机16卡

2机、4机等多机情况下，需要在所有机器节点上相同路径准备同样的数据集和代码脚本，以完成分布式训练。

如2机，NODE1='10.11.0.2'，NODE2='10.11.0.3'，指定batch size=32，每组测试5次，关闭混合精度和xla，则：

在NODE1节点`models/official/nlp/bert/`目录下,执行脚本：

```shell
bash run_two_node.sh 32 5 fp32 false 0
```

在NODE2节点相同目录下，执行同样的脚本：

```shell
bash run_two_node.sh 32 5 fp32 false 1
```

注意：最后一个参数表示task_index，需要与当前节点对应（NODE1为0，NODE2为1）。


## 4机32卡

流程同上，如指定batch size=32，每组测试5次，关闭混合精度和xla，则只需在4个机器节点上分别执行：

```shell
bash run_multi_node.sh 32 5 fp32 false ${task_index}
```

同样，最后一个参数表示task_index，需要与当前节点对应（NODE1为0，NODE2为1，NODE3为2……）。


## 混合精度

可以修改脚本`run_single_node.sh`, `run_two_node.sh`, `run_multi_node.sh`中的变量，或直接通过参数指定DTYPE以开启混合精度，如：

```shell
bash run_single_node.sh 64 5 fp16
```

表示单机训练开启fp16混合精度，并batch size=64，每组测试5次。
多机训练类似，在各节点运行脚本时分别指定对应参数即可。



# Result
## 加速比

执行以下脚本计算各个情况下的加速比：
```shell
python extract_tensorflow_logs_time.py --log_dir=logs/tensorflow/bert/bz64 --batch_size_per_device=64
```
输出：
```shell
logs/tensorflow/bert/bz64/4n8g/bert_b64_fp32_2.log {2: 2237.6}
logs/tensorflow/bert/bz64/4n8g/bert_b64_fp32_1.log {2: 2237.6, 1: 2238.39}
logs/tensorflow/bert/bz64/4n8g/bert_b64_fp32_5.log {2: 2237.6, 1: 2238.39, 5: 2244.38}
logs/tensorflow/bert/bz64/4n8g/bert_b64_fp32_3.log {2: 2237.6, 1: 2238.39, 5: 2244.38, 3: 2252.55}
logs/tensorflow/bert/bz64/4n8g/bert_b64_fp32_4.log {2: 2237.6, 1: 2238.39, 5: 2244.38, 3: 2252.55, 4: 2249.41}
logs/tensorflow/bert/bz64/1n2g/bert_b64_fp32_2.log {2: 202.48}
logs/tensorflow/bert/bz64/1n2g/bert_b64_fp32_1.log {2: 202.48, 1: 204.3}
logs/tensorflow/bert/bz64/1n2g/bert_b64_fp32_5.log {2: 202.48, 1: 204.3, 5: 203.15}
logs/tensorflow/bert/bz64/1n2g/bert_b64_fp32_3.log {2: 202.48, 1: 204.3, 5: 203.15, 3: 204.16}
logs/tensorflow/bert/bz64/1n2g/bert_b64_fp32_4.log {2: 202.48, 1: 204.3, 5: 203.15, 3: 204.16, 4: 204.96}
logs/tensorflow/bert/bz64/1n1g/bert_b64_fp32_2.log {2: 114.95}
logs/tensorflow/bert/bz64/1n1g/bert_b64_fp32_1.log {2: 114.95, 1: 113.55}
logs/tensorflow/bert/bz64/1n1g/bert_b64_fp32_5.log {2: 114.95, 1: 113.55, 5: 111.67}
logs/tensorflow/bert/bz64/1n1g/bert_b64_fp32_3.log {2: 114.95, 1: 113.55, 5: 111.67, 3: 112.99}
logs/tensorflow/bert/bz64/1n1g/bert_b64_fp32_4.log {2: 114.95, 1: 113.55, 5: 111.67, 3: 112.99, 4: 112.71}
logs/tensorflow/bert/bz64/2n8g/bert_b64_fp32_2.log {2: 1242.44}
logs/tensorflow/bert/bz64/2n8g/bert_b64_fp32_1.log {2: 1242.44, 1: 1237.07}
logs/tensorflow/bert/bz64/2n8g/bert_b64_fp32_5.log {2: 1242.44, 1: 1237.07, 5: 1249.61}
logs/tensorflow/bert/bz64/2n8g/bert_b64_fp32_3.log {2: 1242.44, 1: 1237.07, 5: 1249.61, 3: 1237.47}
logs/tensorflow/bert/bz64/2n8g/bert_b64_fp32_4.log {2: 1242.44, 1: 1237.07, 5: 1249.61, 3: 1237.47, 4: 1239.3}
logs/tensorflow/bert/bz64/1n4g/bert_b64_fp32_2.log {2: 402.02}
logs/tensorflow/bert/bz64/1n4g/bert_b64_fp32_1.log {2: 402.02, 1: 399.56}
logs/tensorflow/bert/bz64/1n4g/bert_b64_fp32_5.log {2: 402.02, 1: 399.56, 5: 400.27}
logs/tensorflow/bert/bz64/1n4g/bert_b64_fp32_3.log {2: 402.02, 1: 399.56, 5: 400.27, 3: 404.06}
logs/tensorflow/bert/bz64/1n4g/bert_b64_fp32_4.log {2: 402.02, 1: 399.56, 5: 400.27, 3: 404.06, 4: 402.34}
logs/tensorflow/bert/bz64/1n8g/bert_b64_fp32_2.log {2: 805.43}
logs/tensorflow/bert/bz64/1n8g/bert_b64_fp32_1.log {2: 805.43, 1: 806.74}
logs/tensorflow/bert/bz64/1n8g/bert_b64_fp32_5.log {2: 805.43, 1: 806.74, 5: 803.36}
logs/tensorflow/bert/bz64/1n8g/bert_b64_fp32_3.log {2: 805.43, 1: 806.74, 5: 803.36, 3: 806.01}
logs/tensorflow/bert/bz64/1n8g/bert_b64_fp32_4.log {2: 805.43, 1: 806.74, 5: 803.36, 3: 806.01, 4: 805.41}
{'bert': {'1n1g': {'average_speed': 113.17,
                   'batch_size_per_device': 64,
                   'median_speed': 112.99,
                   'speedup': 1.0},
          '1n2g': {'average_speed': 203.81,
                   'batch_size_per_device': 64,
                   'median_speed': 204.16,
                   'speedup': 1.81},
          '1n4g': {'average_speed': 401.65,
                   'batch_size_per_device': 64,
                   'median_speed': 402.02,
                   'speedup': 3.56},
          '1n8g': {'average_speed': 805.39,
                   'batch_size_per_device': 64,
                   'median_speed': 805.43,
                   'speedup': 7.13},
          '2n8g': {'average_speed': 1241.18,
                   'batch_size_per_device': 64,
                   'median_speed': 1239.3,
                   'speedup': 10.97},
          '4n8g': {'average_speed': 2244.47,
                   'batch_size_per_device': 64,
                   'median_speed': 2244.38,
                   'speedup': 19.86}}}
Saving result to ./result/_result.json
```
## 计算规则

### 1.测速脚本

- extract_tensorflow_logs_time.py

extract_tensorflow_logs_time.py根据log中打印出的时间，排除前20iter取后100个iter的实际运行时间计算速度。

### 2.均值速度和中值速度

- average_speed均值速度

- median_speed中值速度

  每个batch size进行5次训练测试，记为一组，每一组取average_speed为均值速度，median_speed为中值速度

### 3.加速比以中值速度计算

脚本和表格中的 **加速比** 是以单机单卡下的中值速度为基准进行计算的。例如:

单机单卡情况下速度为200(samples/s)，单机2卡速度为400，单机4卡速度为700，则加速比分别为：1.0、2.0、3.5

## BERT-Base FP32

### batch size=32 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 103.58    | 1       |
| 1        | 2       | 177.18    | 1.71    |
| 1        | 4       | 347.83    | 3.36    |
| 1        | 8       | 675.82    | 6.52    |
| 2        | 16      | 909.0     | 8.78    |
| 4        | 32      | 1551.52   | 14.98   |

### batch size=32 & with xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 125.37    | 1       |
| 1        | 2       | 216.46    | 1.73    |
| 1        | 4       | 421.79    | 3.36    |
| 1        | 8       | 775.93    | 6.19    |
| 2        | 16      | 1022.99   | 8.16    |
| 4        | 32      | 1693.59   | 13.51   |

### batch size=48 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 108.94    | 1       |
| 1        | 2       | 194.29    | 1.78    |
| 1        | 4       | 384.59    | 3.53    |
| 1        | 8       | 752.21    | 6.9     |
| 2        | 16      | 1106.71   | 10.16   |
| 4        | 32      | 1956.1    | 17.96   |

### batch size=48 & with xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 131.27    | 1       |
| 1        | 2       | 236.58    | 1.8     |
| 1        | 4       | 468.83    | 3.57    |
| 1        | 8       | 877.55    | 6.69    |
| 2        | 16      | 1262.4    | 9.62    |
| 4        | 32      | 2168.44   | 16.52   |

### batch size = 64 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 112.99    | 1       |
| 1        | 2       | 204.16    | 1.81    |
| 1        | 4       | 402.02    | 3.56    |
| 1        | 8       | 805.43    | 7.13    |
| 2        | 16      | 1239.3    | 10.97   |
| 4        | 32      | 2244.38   | 19.86   |

### batch size = 64 & with xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 138.21    | 1       |
| 1        | 2       | 254.91    | 1.84    |
| 1        | 4       | 505.63    | 3.66    |
| 1        | 8       | 959.02    | 6.94    |
| 2        | 16      | 1448.74   | 10.48   |
| 4        | 32      | 2544.74   | 18.41   |

注：
- 以32为最小单位，最大batch size为64，否则会OOM(out of memory)。


## BERT-Base FP16

### batch size=64 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 228.66    | 1       |
| 1        | 2       | 385.19    | 1.68    |
| 1        | 4       | 746.9     | 3.27    |
| 1        | 8       | 1402.41   | 6.13    |
| 2        | 16      | 1893.51   | 8.28    |
| 4        | 32      | 3194.02   | 13.97   |

### batch size=64 & with xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 414.78    | 1       |
| 1        | 2       | 748.66    | 1.8     |
| 1        | 4       | 1311.35   | 3.16    |
| 1        | 8       | 2241.6    | 5.4     |
| 2        | 16      | 2565.99   | 6.19    |
| 4        | 32      | 4113.68   | 9.92    |

### batch size=96 & without xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 258.18    | 1       |
| 1        | 2       | 440.72    | 1.71    |
| 1        | 4       | 868.12    | 3.36    |
| 1        | 8       | 1669.07   | 6.46    |
| 2        | 16      | 2421.33   | 9.38    |
| 4        | 32      | 4168.79   | 16.15   |

### batch size=96 & with xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 466.34    | 1       |
| 1        | 2       | 851.97    | 1.83    |
| 1        | 4       | 1557.31   | 3.34    |
| 1        | 8       | 2773.8    | 5.95    |
| 2        | 16      | 3506.12   | 7.52    |
| 4        | 32      | 5735.19   | 12.3    |

### batch size=128 & with xla

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       | 503.47    | 1       |
| 1        | 2       | 924.4     | 1.84    |
| 1        | 4       | 1694.3    | 3.37    |
| 1        | 8       | 3200.86   | 6.36    |
| 2        | 16      | 4243.96   | 8.43    |
| 4        | 32      | 7101.16   | 14.1    |

注：
- 以32为最小单位，关闭xla时，最大batch size为96，否则会OOM(out of memory)。
- 以32为最小单位，打开xla时，最大batch size为128，否则会OOM(out of memory)。


## 完整日志

- [bert_fp32.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/Tensorflow/bert/bert_fp32.zip)
- [bert_fp32_xla.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/Tensorflow/bert/bert_fp32_xla.zip)
- [bert_fp16.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/Tensorflow/bert/bert_fp16.zip)
- [bert_fp16_xla.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/Tensorflow/bert/bert_fp16_xla.zip)

