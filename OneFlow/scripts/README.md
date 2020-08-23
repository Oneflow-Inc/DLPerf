# OneFlow Benchmark Test Scripts
本文介绍如何使用脚本批量测试ResNet50 V1.5和BERT base，涉及的脚本包括：
1. `rn50_train.sh`，可以单机本地进行训练resnet 50，也可以通过ssh发送到远端节点运行；
2. `bert_base_train.sh`，可以单机本地进行BERT base预训练，也可以通过ssh发送到远端节点运行；
3. `launch_all.sh`，发送脚本到指定的机器节点并运行；
4. `cp_logs.sh`，拷贝日志到指定目录；
5. `schedule_launch.sh`，批量顺序执行多组`launch_all.sh`；
6. `extract_bert_result.py`，从BERT预训练日志中提取结果，并打印成markdown表格。
7. `extract_cnn_result.py`，从cnn训练日志中提取结果，并打印成markdown表格。

通常这几个文件需要修改很少的配置才能够正常运行，下面对各个脚本进行详细介绍。
## 本地训练启动脚本：`rn50_train.sh`和`bert_base_train.sh`
这两个脚本用于本地运行oneflow的训练，可以独立使用，调用前需要手动修改基本配置，调用时需要传入3个参数。
### 手工配置选项
手工配置选项以ResNet50为例，有三处需要修改的地方：
```
BENCH_ROOT=cnns
DATA_ROOT=/path/to/imagenet_ofrecord
DATA_PART_NUM=32
```
1. `BENCH_ROOT`: 模型脚本所在的目录，对应OneFlow-Benchmark项目中的`Classification/cnns`目录或`LanguageModeling/BERT`；
2. `DATA_ROOT`: 测试用数据集所在的目录
3. `DATA_PART_NUM`: 测试用数据集文件数量
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
4. `NODE_IPS`: 各个节点的IP列表，可选。

注：这两个脚本只能够在本地运行OneFlow，如果多机训练，可以在各台机器分别启动该脚本，OneFlow会自动根据配置的机器节点信息进行通信连接完成训练。另外一种方式就是使用`launch_all.sh`自动把脚本发送到各个机器节点进行训练。

调用示例，本地单机8卡训练ResNet50:
```
./rn50_train.sh 1 8 128
```

## 远程训练启动脚本：`launch_all.sh`
`launch_all.sh`负责发送本地训练启动脚本和模型脚本目录到各台机器节点，并通过ssh的方式在各个机器节点运行本地训练启动脚本。启动时，需要传入5个参数：
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


调用示例1，发送到单机使用8卡训练ResNet50:
```
./launch_all.sh rn50_train.sh cnns 1 8 128
```

调用示例2，发送到4机每机都使用8卡训练BERT base:
```
./launch_all.sh bert_base_pretrain.sh BERT 1 8 96
```

## `cp_logs.sh`
根据下列参数拷贝日志到指定目录并重命名：
```
NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BSZ=$3
REPEAT_ID=$4
```
`launch_all.sh`会在远程机器上建立一个`oneflow_temp`的文件夹，并且把脚本都scp到该文件夹中，训练时日志也被保存在`oneflow_temp`里，`cp_logs.sh`负责从主节点的`oneflow_temp`目录中拷贝日志。

## `schedule_launch.sh`
根据需求批量自动运行`launch_all.sh`和`cp_logs.sh`，完成训练和备份日志。
需要两个参数：
1. `LOCAL_RUN`：待发送的本地训练启动脚本；
2. `BENCH_ROOT`: 待发送的OneFlow模型脚本所在目录。

在当前的`schedule_launch.sh`脚本中会循环不同batch size，4组节点和GPU设备数量，每组实验重复7次，每次实验结束后日志被拷贝到当前目录的`logs/oneflow`目录下。

## `extract_result.py`
由于ResNet50和BERT base的日志格式有所不同，所以有两个`extract_result.py`文件分别放在不同的目录里。
运行方式如下：
```
python3 BERT_base/extract_result.py
```
生成结果直接被打印到屏幕，是markdown格式方便直接引用，如下
```
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
| -------- | -------- | -------- | -------- |
| 4 | 8 | 96 | 4449.9 |
| 4 | 8 | 96 | 4456.8 |
| 4 | 8 | 96 | 4460.2 |
| 4 | 8 | 96 | 4455.0 |
| 4 | 8 | 96 | 4456.0 |
| 4 | 8 | 96 | 4451.4 |
| 4 | 8 | 96 | 4458.1 |
```