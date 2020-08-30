# OneFlow Benchmark Test Scripts

本文介绍如何使用脚本批量测试ResNet50 V1.5和BERT base：

1. `rn50_train.sh`，可以本地单机训练resnet 50，也可以通过ssh发送到远端节点运行；
2. `bert_base_train.sh`，可以本地单机进行BERT base预训练，也可以通过ssh发送到远端节点运行；
3. `launch_all.sh`，发送脚本到指定的机器节点并运行；
4. `cp_logs.sh`，拷贝日志到指定目录；
5. `schedule_launch.sh`，批量顺序执行多组`launch_all.sh`；
6. `extract_bert_result.py`，从BERT预训练日志中提取结果，并打印成markdown表格。
7. `extract_cnn_result.py`，从cnn训练日志中提取结果，并打印成markdown表格。

通常这几个文件只需要修改很少的配置就能正常运行，下面对各个脚本进行详细介绍。

## 本地训练启动脚本：`rn50_train.sh`和`bert_base_train.sh`

这两个脚本用于本地运行OneFlow的训练，可以独立使用，调用前需要手动修改基本配置，调用时需要传入3个参数。

### 手工配置选项

手工配置选项以ResNet50为例，有三处需要修改的地方：

```
BENCH_ROOT=cnns
DATA_ROOT=/path/to/imagenet_ofrecord
DATA_PART_NUM=32
```

1. `BENCH_ROOT`: 模型脚本所在的目录，对应OneFlow-Benchmark项目中的`Classification/cnns`目录或`LanguageModeling/BERT`；
2. `DATA_ROOT`: 测试所用数据集路径
3. `DATA_PART_NUM`: 测试所用数据集文件数量

### 脚本参数

调用时需要传入4个参数：

```
NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BSZ_PER_DEVICE=$3
NODE_IPS=$4
```

1. `NUM_NODES`: 测试训练用的机器节点数量；
2. `GPU_NUM_PER_NODE`: 每台机器节点中GPU设备的数量；
3. `BSZ_PER_DEVICE`: 训练时每个批次每个GPU设备对应的图片/句子数量；
4. `NODE_IPS`: 各个节点的IP列表，可选，多机训练须配置（如果NUM_NODES=1，则NODE_IPS被忽略）。

注：这两个脚本只能够在本地运行OneFlow，如果多机训练，可以在各台机器上分别启动该脚本，OneFlow会自动根据配置的机器节点信息进行通信连接，完成训练。另外一种方式就是使用`launch_all.sh`，自动把脚本发送到各个机器节点进行训练。

本地单机8卡训练ResNet50，执行命令:

```
./rn50_train.sh 1 8 128
```

## 远程训练启动脚本：`launch_all.sh`

`launch_all.sh`负责发送本地训练启动脚本（如单机训练脚本rn50_train.sh）和`BENCH_ROOT`路径下的模型脚本（如Classification/cnns/of_cnn_train_val.py等）到各台机器节点，并通过ssh的方式在各个机器节点运行本地训练启动脚本。启动时，需要传入5个参数：

```
LOCAL_RUN=$1
BENCH_ROOT=$2
NUM_NODES=$3
GPU_NUM_PER_NODE=$4
BSZ=$5
```

1. `LOCAL_RUN`：待发送的本地训练启动脚本；
2. `BENCH_ROOT`: 待发送的OneFlow模型脚本所在目录；
3. `NUM_NODES`: 测试训练用的机器节点数量；
4. `GPU_NUM_PER_NODE`: 每台机器节点中GPU设备的数量；
5. `BSZ_PER_DEVICE`: 训练时每个批次每个GPU设备对应的图片/句子数量；


发送相关脚本到单机，使用8卡训练ResNet50:

```
./launch_all.sh rn50_train.sh cnns 1 8 128
```

发送相关脚本到4机，每机都使用8卡（共4机32卡）训练BERT base:

```
./launch_all.sh bert_base_pretrain.sh BERT 4 8 96
```

## `cp_logs.sh`

根据下列参数拷贝日志到指定路径并重命名：

```
NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BSZ=$3
REPEAT_ID=$4
```

`cp_logs.sh`负责从本地（主节点）拷贝日志到指定路径下，并按照`logs/oneflow/${NUM_NODES}n${GPU_NUM_PER_NODE}g/${model_name}_b${BSZ}_fp32_${REPEAT_ID}.log`的格式保存。

## `schedule_launch.sh`

本次测评会测试多组batch_size、单机多机配置，每组实验重复7次。

根据测试次数，批量自动运行`launch_all.sh`和`cp_logs.sh`，完成训练和备份日志。
需要两个参数：

1. `LOCAL_RUN`：待发送的本地训练启动脚本；
2. `BENCH_ROOT`: 待发送的OneFlow模型脚本所在目录。

`schedule_launch.sh`脚本会根据实验次数，循环测试不同batch size，4组节点和GPU设备数量，每组实验重复7次。实验结束后，`logs/oneflow`路径下会保存实验日志。

## `extract_bert_result.py` `extract_cnn_result.py`

由于ResNet50和BERT base的日志格式有所不同，所以有两个提取脚本，以bert为例，运行方式如下：

```
python3 extract_bert_result.py
```

结果为markdown格式，方便直接引用，输出如下:

```
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
| -------- | -------- | -------- | -------- |
| 4 | 8 | 96 | 4449.85 |
| 4 | 8 | 96 | 4456.82 |
| 4 | 8 | 96 | 4460.17 |
| 4 | 8 | 96 | 4454.99 |
| 4 | 8 | 96 | 4455.97 |
| 4 | 8 | 96 | 4451.41 |
| 4 | 8 | 96 | 4458.06 |


| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
| -------- | -------- | -------- | -------- | -------- |
| 1 | 1 | 96 | 149.84 | 1.00 |
| 1 | 8 | 96 | 1158.51 | 7.73 |
| 2 | 8 | 96 | 2257.71 | 15.07 |
| 4 | 8 | 96 | 4455.97 | 29.74 |
```

### 输入参数

- `benchmark_log_dir`: 日志存放的目录，脚本中会自动遍历所有`*.log`文件进行信息提取；
- `start_iter` `end_iter`: 待提取的起始和终止步数，脚本中会利用这两个步数的时间戳计算吞吐率。
- `print_mode`: 打印输出格式设置，缺省`markdown`
