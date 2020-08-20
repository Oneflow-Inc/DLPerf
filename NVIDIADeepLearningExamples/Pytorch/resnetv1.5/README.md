# ResNet50 

## 1. 环境

### 系统信息

- 系统：Ubuntu 16.04
- 显卡：Tesla V100（16G）×8
- 驱动：440.33.01
- CUDA：10.2
- cudnn：7.6.5
- nccl：2.7.3
- pytorch版本：1.6.0

### 项目代码

pytorch官方仓库：

https://github.com/pytorch/examples

```
# install pytorch
python -m pip install torch==1.6.0 torchvision==0.7.0  -i https://mirror.baidu.com/pypi/simple
```

由于此次是训练性能测试，故注释掉 evaluate 相关的代码，具体位于：`pytorch/examples/imagenet/main.py  `239行：

```
# # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion, args)

        # # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best)
```

## 2. 数据处理

数据需要预先处理，下载脚本：[📎valprep.sh](https://www.yuque.com/attachments/yuque/0/2020/sh/216914/1597242115544-336be8b6-e448-420d-a688-b2fc2c61d535.sh)(脚本来源：[pytorch官方仓库](https://github.com/pytorch/examples/tree/master/imagenet))

将脚本放入val文件夹后执行`sh valprep.sh` 将图片按照分类放入文件夹中，处理后的验证集格式如下：

```
.
├── n01440764
├── n01443537
├──      ...
└── valprep.sh
```

测速时我们只关注训练，忽略验证过程，故此处只准备一个验证文件夹用于跑通流程即可，如：n01440764

## 3. 运行

### 单机运行

集群中有4台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有8张显卡，这里设置batch_size=128，从1机1卡～4机32卡进行了一组训练

## 1机1卡

**train.sh**

`pytorch/examples/imagenet`下，新建train.sh

脚本内容如下：

```
#!/bin/sh
MODEL=${1:-"resnet50"}
gpus=${2:-"0"}
bz_per_device=${3:-128}

a=`expr ${#gpus} + 1`
NUM_GPU=`expr ${a} / 2`
total_bz=`expr ${bz_per_device} \* ${NUM_GPU}`
LR=$(awk -v total_bz="$total_bz" 'BEGIN{print  total_bz / 1000}')
echo "Use gpus: $gpus"
echo "Batch size : $bz_per_device"
echo "Learning rate: $LR"

# LOG_FOLDER=$OUTPUT_DIR/pytorch/benchmark_log/cnns/${MODEL}_1node_bz${bz_per_device}
LOG_FOLDER=../logs/benchmark_log/cnns/${MODEL}_1node_bz${bz_per_device}
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/1n${NUM_GPU}c_real.log


export CUDA_VISIBLE_DEVICES=${gpus}
DATA_DIR=/datasets/ImageNet/imagenet_1k/

python3 main.py $DATA_DIR \
        --arch $MODEL  \
        --epochs 1 \
        --batch-size $total_bz \
        --lr  $LR \
        --momentum 0.9 \
        --print-freq 1  2>&1 | tee ${LOGFILE}
        
echo "Writting log to ${LOGFILE}"
```

执行`sh train.sh`即可运行训练。

### 进阶

#### *多机运行*

- 2机16卡

多节点的脚本和单机脚本稍微有些不同，在多机脚本中需要设置所有参与训练的节点ip，当前节点ip。集群中有4台节点，我们选择任意2个节点参与训练，这里选择1和2号机器节点。



在1号节点`pytorch/examples/imagenet`下新建`distributed_train.sh`:

```
#!/bin/sh
NODE1=10.11.0.2:11111   
NODE2=10.11.0.3:11111     
NODE3=10.11.0.4:11111     
NODE4=10.11.0.5:11111
MODEL=${1:-"resnet50"}
gpus=${2:-"0,1,2,3,4,5,6,7"}
bz_per_device=${3:-128}
# node_ips=${4:-$NODE1,$NODE2,$NODE3,$NODE4}
node_ips=${4:-$NODE1,$NODE2}
master_node=$NODE1


a=`expr ${#gpus} + 1`
NUM_GPU=`expr ${a} / 2`
total_bz=`expr ${bz_per_device} \* ${NUM_GPU}`
LR=$(awk -v total_bz="$total_bz" 'BEGIN{print  total_bz / 1000}')
node_num=$(echo $node_ips | tr ',' '\n' | wc -l)
echo "Use gpus: $gpus"
echo "Batch size : $bz_per_device"
echo "Learning rate: $LR"

# LOG_FOLDER=$OUTPUT_DIR/pytorch/benchmark_log/cnns/${MODEL}_1node_bz${bz_per_device}
LOG_FOLDER=../logs/benchmark_log/cnns/${MODEL}_${node_num}node_bz${bz_per_device}
mkdir -p $LOG_FOLDER
LOGFILE=${LOG_FOLDER}/${node_num}n${NUM_GPU}c_real.log


export CUDA_VISIBLE_DEVICES=${gpus}
DATA_DIR=/datasets/ImageNet/imagenet_1k/

python3 main.py  $DATA_DIR  \
        --arch $MODEL \
        --epochs 1 \
        --batch-size $total_bz \
        --lr $LR \
        --momentum 0.9 \
        --print-freq 1 \
        --multiprocessing-distributed \
        --dist-backend  nccl \
        --dist-url  tcp://$master_node \
        --world-size $node_num \
        --rank 1 2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"
```

在2号节点下同样新建脚本： `distributed_train.sh`  脚本内容和1号节点相同，只需修改参数：`--rank 1`即可。

在1、2号节点上分别执行：`sh distributed_train.sh`



#### 优化

【。。。】



#### 更多参数

进行参数的更多讲解

