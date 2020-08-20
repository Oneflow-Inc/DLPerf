# Pytorch 框架测试

简介框架背景，参考框架链接

## 1. 使用说明 Usage

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

## 2. 环境 Environmrnt

### 系统信息

- 硬件
  - 显卡：Tesla V100（16G）×8
- 软件
  - 系统：Ubuntu 16.04
  - 驱动：440.33.01
  - CUDA：10.2
  - cudnn：7.6.5
  - nccl：2.7.3
  - pytorch版本：1.6.0



## 3. 性能 Performance

本结果是基于第 2 节中的软硬件环境配置进行测试，测试结果如下：

| 节点-GPU | samples/s | 加速比 |
| -------- | --------- | ------ |
| 1机1卡   | 347.12    | 1      |
| 1机2卡   | 604.11    | 1.74   |
| 1机4卡   | 899.41    | 2.59   |
| 1机8卡   | 1404.55   | 4.05   |
| 2机16卡  | 2384.3    | 6.87   |
| 4机32卡  | 5552.77   | 16.0   |

###  

### ResNet50 V1.5 bsz = 160

| Node num | GPU num | OneFlow   | TensorFlow | PyTorch | MxNet | Paddle   |
| -------- | ------- | --------- | ---------- | ------- | ----- | -------- |
| 1        | 1       | 383.760   | 336.23     | 350.63  | TODO  | 353.68   |
| 1        | 2       | 747.295   | 585.5      | 614.82  | TODO  | 687.09   |
| 1        | 4       | 1497.618  | 1142.71    | 928.19  | TODO  | 1065.53  |
| 1        | 8       | 2942.321  | 2626.03    | 1079.27 | TODO  | 1442.32  |
| 2        | 16      | 5839.054  | OOM        | 2386.48 | TODO  | 5128.92  |
| 4        | 32      | 11548.451 | 9869.94    | 5594.45 | TODO  | 10087.09 |

###  

### ResNet50 V1.5 bsz = 128

| Node num | GPU num | OneFlow   | TensorFlow | PyTorch | MxNet | Paddle  |
| -------- | ------- | --------- | ---------- | ------- | ----- | ------- |
| 1        | 1       | 383.760   | 326.96     | 347.12  | TODO  | 353.66  |
| 1        | 2       | 747.295   | 590.94     | 604.11  | TODO  | 681.4   |
| 1        | 4       | 1497.618  | 1164.24    | 899.41  | TODO  | 1093.7  |
| 1        | 8       | 2942.321  | 2513.72    | 1404.55 | TODO  | 1302.12 |
| 2        | 16      | 5839.054  | 4752.52    | 2384.3  | TODO  | 5161.04 |
| 4        | 32      | 11548.451 | 9407.87    | 5552.77 | TODO  | 9995.53 |



### ResNet50 V1.5 bsz = 64

| Node num | GPU num | OneFlow | TensorFlow | PyTorch | MxNet | Paddle  |
| -------- | ------- | ------- | ---------- | ------- | ----- | ------- |
| 1        | 1       | TODO    | 273.32     | 337.1   | TODO  | 339.69  |
| 1        | 2       | TODO    | 528.31     | 572.22  | TODO  | 629.58  |
| 1        | 4       | TODO    | 1022.4     | 845.79  | TODO  | 1005.67 |
| 1        | 8       | TODO    | 2146.8     | 1409.76 | TODO  | 1176.99 |
| 2        | 16      | TODO    | 4129.19    | 2246.7  | TODO  | 4499.43 |
| 4        | 32      | TODO    | 7818.74    | 4590.78 | TODO  | 8661.84 |



### ResNet50 V1.5 bsz = 32

| Node num | GPU num | OneFlow | TensorFlow | PyTorch | MxNet | Paddle  |
| -------- | ------- | ------- | ---------- | ------- | ----- | ------- |
| 1        | 1       | TODO    | 217.57     | 306.82  | TODO  | 311.51  |
| 1        | 2       | TODO    | 467.5      | 467.56  | TODO  | 526.54  |
| 1        | 4       | TODO    | 891.21     | 719.73  | TODO  | 921.58  |
| 1        | 8       | TODO    | 1661.2     | 934.23  | TODO  | 1089.29 |
| 2        | 16      | TODO    | 3202.26    | 2188.57 | TODO  | 3540.63 |
| 4        | 32      | TODO    | 5930.8     | 4254.55 | TODO  | 6734.42 |

