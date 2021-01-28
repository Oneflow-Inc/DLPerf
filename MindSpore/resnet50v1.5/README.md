# 【DLPerf】MindSpore-ResNet50v1.5 测评

# Overview

本次复现采用了[MindSpore官方仓库](https://gitee.com/mindspore/mindspore/tree/r1.1)中的[ResNet](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/resnet)，目的在于速度测评，同时根据测速结果给出1机、2机、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试已覆盖 FP32、FP16混合精度，后续将持续维护，增加更多方式的测评。


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

## Feature support matrix

| Feature            | ResNet50v1.5 MindSpore    |
| ------------------ | ------------------------- |
| Mpi Multi-gpu      | Yes                       |
| Mpi Multi-node     | Yes                       |
| Automatic mixed precision (AMP) | Yes                       |

# Quick Start

## 项目代码

- [MindSpore官方仓库](https://gitee.com/mindspore/mindspore/tree/r1.1)
  - [ResNet项目主页](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/resnet)

下载官方源码：

```shell
git clone https://gitee.com/mindspore/mindspore.git
cd mindspore/
git checkout r1.1
cd model_zoo/official/cv/resnet/
```

1.将本页面scripts路径下的脚本：`run_single_node.sh`、`run_multi_node.sh`放入model_zoo/official/cv/resnet/路径下；

2.将本页面scripts路径下的其余脚本：`run_standalone_pretrain_for_gpu.sh`、`run_distributed_pretrain_for_gpu.sh`放入model_zoo/official/cv/resnet/scripts/下；

3.修改代码脚本，将本页面`train.py`放入model_zoo/official/cv/resnet/路径下；

或者按如下说明手动修改：

将 model_zoo/official/cv/resnet/train.py 46 行：
```shell
    # line 46
    args_opt = parser.parse_args()
```
替换为：
```shell
    # line 46
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--train_steps", type=int, default=-1, help="Training Steps, default is -1, "
                                                                    "meaning run all steps according to epoch number.")
    parser.add_argument("--data_sink_steps", type=int, default="1", help="Sink steps for each epoch, default is 1.")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="dtype, default is fp32.")

    args_opt = parser.parse_args()
    from mindspore import log as logger
    logger.warning("\nargs_opt: {}".format(args_opt))
```
以增加输入参数。

将 model_zoo/official/cv/resnet/train.py 81 行：
```shell
    # line 81
    ckpt_save_dir = config.save_checkpoint_path
```
替换为：
```shell
    # line 81
    config.save_checkpoint = False
    config.epoch_size = 1
    config.batch_size = args_opt.batch_size
    config.momentum = 0.875
    config.lr_init = 0.001
    amp_level = "O2" if args_opt.dtype == "fp16" else "O0"
    ckpt_save_dir = config.save_checkpoint_pat
```
以设置测试参数。

将 model_zoo/official/cv/resnet/train.py 187 行至 190 行：
```shell
    # line 187
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    # Mixed precision
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False)
```
替换为：
```shell
    from mindspore.train.loss_scale_manager import DynamicLossScaleManager
    loss_scale = DynamicLossScaleManager(2,1,10)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                amp_level=amp_level, keep_batchnorm_fp32=False)
```
以支持设置混合精度和 dynamic loss scale。

将
(1) model_zoo/official/cv/resnet/train.py 46 行：
```shell
    # line 197
    time_cb = TimeMonitor(data_size=step_size)
```
替换为：
```shell
    # line 197
    time_cb = TimeMonitor(data_size=args_opt.data_sink_steps)
```
(2) model_zoo/official/cv/resnet/train.py 210 行至 211 行：
```shell
    # line 210
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)
```
替换为：
```shell
    # line 210
    assert(args_opt.train_steps <= dataset.get_dataset_size())
    train_steps = dataset.get_dataset_size() if args_opt.train_steps < 0 else args_opt.train_steps
    new_repeat_count = (config.epoch_size - config.pretrain_epoch_size) * train_steps // args_opt.data_sink_steps
    model.train(new_repeat_count, dataset, callbacks=cb,
                sink_size=args_opt.data_sink_steps, dataset_sink_mode=dataset_sink_mode)
```
以设置训练 step 和输出时间信息的间隔，便于统计性能。

## 容器

本次测评采用的是MindSpore官方提供的Docker镜像，您可以
参考[MindSpore官方文档](https://gitee.com/mindspore/mindspore/tree/r1.1/#docker%E9%95%9C%E5%83%8F)GPU部分
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
    --name mindspore_resnet \
    -v /dev/shm:/dev/shm \
    -v $PWD:/workspace/resnet \
    -v /home/leinao/dataset/ImageNet:/workspace/resnet/data/ImageNet \
    -v $PWD/results:/results \
    mindspore/mindspore-gpu:1.1.0 /bin/bash
```

## 数据集

数据集直接采用 JPEG 图像，请参考：[MindSpore官方仓库说明](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/resnet#%E6%95%B0%E6%8D%AE%E9%9B%86) ImageNet2012 部分；

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

每个节点有8张显卡，这里设置batch_size=128，从1机1卡～4机32卡进行了一组训练

## 单机

进入容器：

```shell
docker exec -it mindspore_resnet /bin/bash
cd /workspace/resnet
bash run_single_node.sh 128 fp32 5
```

执行脚本测试fp32+batch size128，对单机1卡、2卡、4卡、8卡分别做5次测试，也可以指定其他batch_size参数进行测试。

### 混合精度

指定dtype运行参数为fp16，以进行fp16混合精度测试，例如：

- batch size=256 使用fp16混合精度：

```shell
bash   run_single_node.sh 256 fp16 5
```

## 多机

2机、4机等多机情况下，需要在所有机器节点上相同路径准备同样的数据集、以完成分布式训练。由于配置了ssh免密，您只需要在一个节点上运行脚本即可执行多机训练。

如2机：NODE1='10.11.0.2'   NODE2='10.11.0.3' 的训练，需在两台机器上分别准备好数据集后，NODE1节点进入容器/workspace/resnet下，执行脚本:

```shell
bash   run_multi_node.sh 128 fp32 5 2
```
即可运行2机16卡的训练，同样测试5次。

### 混合精度

指定dtype运行参数为fp16，以进行fp16混合精度测试，例如：

- batch size=256 使用fp16混合精度：

```shell
bash   run_multi_node.sh 256 fp16 5 2
```

## Result

### 吞吐率及加速比

执行以下命令，即可计算各种测试配置下的吞吐率及加速比：

```shell
python extract_mindspore_logs_time.py --log_dir=logs_fp32/mindspore/resnet50/bz128
```

输出：

```shell
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

### ResNet50v1.5  FP32

#### batch size=128

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       |    |    |
| 1        | 2       |    |    |
| 1        | 4       |    |    |
| 1        | 8       |    |    |
| 2        | 16      |    |    |
| 4        | 32      |    |    |

### ResNet50v1.5  FP16

#### batch size=256

| node_num | gpu_num | samples/s | speedup |
| -------- | ------- | --------- | ------- |
| 1        | 1       |    |    |
| 1        | 2       |    |    |
| 1        | 4       |    |    |
| 1        | 8       |    |    |
| 2        | 16      |    |    |
| 4        | 32      |    |    |

### 完整日志

- [resnet50_fp32.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MindSpore/resnet50/resnet50_fp32.zip) 
- [resnet50_fp16.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/MindSpore/resnet50/resnet50_fp16.zip) 
