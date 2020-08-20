# 【Benchmark-TF1.5】NVIDIA-ResNet50测评

# 环境
## 系统信息

- 系统：Ubuntu 16.04
- 显卡：Tesla V100（16G）×8
- 驱动：440.33.01
- CUDA：10.2
- cudnn：7.6.5
- nccl：2.7.3
- horovod：0.19.5
- openmpi：4.0.0
### 容器

- Ubuntu18.04
- Python 3.6
- Tensorflow 1.15.2
- CUDA 10.2.89
- CUDNN 7.6.5
- NCCL 2.6.3
- Horovod 0.19.0
- OpenMPI 3.1.4
- DALI 0.19.0



## 项目代码

- [NVIDIA官方仓库](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590)
   - [Resnet50_v1.5项目主页](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5)



```shell
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples && checkout fed7ba99cde958fda12c9e81d12b3d7e738e0590
```


下载脚本：[scripts.zip](https://www.yuque.com/attachments/yuque/0/2020/zip/216914/1597917804892-d585ccb5-d418-4b3a-abf9-1c77dd96a38b.zip?_lake_card=%7B%22uid%22%3A%221597917804802-0%22%2C%22src%22%3A%22https%3A%2F%2Fwww.yuque.com%2Fattachments%2Fyuque%2F0%2F2020%2Fzip%2F216914%2F1597917804892-d585ccb5-d418-4b3a-abf9-1c77dd96a38b.zip%22%2C%22name%22%3A%22scripts.zip%22%2C%22size%22%3A3757%2C%22type%22%3A%22application%2Fzip%22%2C%22ext%22%3A%22zip%22%2C%22progress%22%3A%7B%22percent%22%3A99%7D%2C%22status%22%3A%22done%22%2C%22percent%22%3A0%2C%22id%22%3A%22my31a%22%2C%22card%22%3A%22file%22%7D) 将脚本：single_node_train.sh和multi_node_train.sh放入
DeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5/training下；将
SINGLE_NODE_RN50_FP32_1E.sh、TWO_NODE_RN50_FP32_1E.sh和MULTI_NODE_RN50_FP32_1E.sh
放入resnet50v1.5/training/FP32目录下


## NGC容器
nvidia ngc官网：[https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
oss下载：
```shell
# 下载镜像
wget xxxxx nvidia_rn50_tf_20.03.tar.gz
# 解压
gunzip nvidia_rn50_tf_20.03.tar.gz
# load镜像
docker load -i nvidia_rn50_tf_20.03.tar
```
修改DeepLearningExamples/TensorFlow/Classification/ConvNets/下的Dockerfile，将：
`ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.03-tf1-py3`改为：`ARG FROM_IMAGE_NAME=fdc4e72f4c15`（将从网络镜像构建改为从本地已有镜像构建）
```shell
# 构建项目镜像 
# DeepLearningExamples/TensorFlow/Classification/ConvNets目录下
docker build . -t nvidia_rn50_tf:20.03-resnet
# 启动容器
docker  run -it --shm-size=16g --ulimit memlock=-1 --privileged  \
--name tf_resnet  --rm \
--cap-add=IPC_LOCK --device=/dev/infiniband \
-v /datasets/ImageNet/tfrecord:/data/tfrecords \
-p 12345:22 \
-d nvidia_rn50_tf:20.03-resnet
```
**
### 配置ssh(可选)
单机情况下无需配置ssh服务，需要测试2机、4机等多机情况下，则需要安装docker容器间的ssh服务，配置ssh免密登录，保证分布式horovod/mpi脚本运行时可以在单机上与其他节点互联
**安装ssh服务端**
```shell
docker exec -it tf_resnet_lyon /bin/bash
apt-get update
apt-get install openssh-server
```
**设置免密登录**

- 节点间：/root/.ssh/id_rsa.pub 互相放到/root/.ssh/authorized_keys中；
- `vim /etc/ssh/sshd_config`[sshd_config.zip](https://www.yuque.com/attachments/yuque/0/2020/zip/216914/1597851226295-2cbdce72-2b3e-4a7b-94c8-98901f654a35.zip?_lake_card=%7B%22uid%22%3A%221597851226137-0%22%2C%22src%22%3A%22https%3A%2F%2Fwww.yuque.com%2Fattachments%2Fyuque%2F0%2F2020%2Fzip%2F216914%2F1597851226295-2cbdce72-2b3e-4a7b-94c8-98901f654a35.zip%22%2C%22name%22%3A%22sshd_config.zip%22%2C%22size%22%3A1708%2C%22type%22%3A%22application%2Fzip%22%2C%22ext%22%3A%22zip%22%2C%22progress%22%3A%7B%22percent%22%3A99%7D%2C%22status%22%3A%22done%22%2C%22percent%22%3A0%2C%22id%22%3A%22eWh97%22%2C%22card%22%3A%22file%22%7D)
- `service ssh restart`
# 数据集
采用imagenet制作的tfrecord格式：train-00000-of-01024 ...train-00015-of-01024共16384张训练集图片。
进入容器，制作dali数据集索引：
```shell
docker exec -it tf_resnet /bin/bash
cd /workspace/rn50v15_tf && mkdir /data/dali_idx
bash ./utils/dali_index.sh /data/tfrecords /data/dali_idx
```
# 训练
集群中有4台节点：


- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5



每个节点有8张显卡，这里设置batch_size=128，从1机1卡～4机32卡进行了一组训练
## 单机
容器/workspace/rn50v15下执行：
`bash ./resnet50v1.5/training/FP32/SINGLE_NODE_RN50_FP32_1E.sh`即可运行训练。
脚本默认会对单机1卡、4卡、8卡分别做6组测试。
```shell
WORKSPACE=${1:-"/workspace/rn50v15_tf"}
DATA_DIR=${2:-"/data"}

i=1
while [ $i -le 6 ]
do
  USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE} ${DATA_DIR}  1 120 128 fp32 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done


i=1
while [ $i -le 6 ]
do
  USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE} ${DATA_DIR}  4 120 128 fp32 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done

i=1
while [ $i -le 6 ]
do
  USE_DALI=1  bash ${WORKSPACE}/resnet50v1.5/training/single_node_train.sh ${WORKSPACE} ${DATA_DIR}  8 120 128 fp32 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
```



## 2机16卡
容器/workspace/rn50v15下执行：
`bash resnet50v1.5/training/FP32/TWO_NODE_RN50_FP32_1E.sh`
即可运行2机16卡的训练，同样默认测试6组。
```shell
WORKSPACE=${1:-"/workspace/rn50v15_tf"}
DATA_DIR=${2:-"/data"}

GPUS_PER_NODE=8
NODE1=10.11.0.2:$GPUS_PER_NODE
NODE2=10.11.0.3:$GPUS_PER_NODE

i=1
while [ $i -le 6 ]
do
  USE_DALI=1   bash ${WORKSPACE}/resnet50v1.5/training/multi_node_train.sh ${WORKSPACE} ${DATA_DIR} \
  $GPUS_PER_NODE 120 128 fp32  $NODE1,$NODE2 1$i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 20
done
```


## 4机32卡
容器/workspace/rn50v15下执行：
`bash resnet50v1.5/training/FP32/MULTI_NODE_RN50_FP32_1E.sh`
即可运行4机32卡的训练，默认测试6组、
```shell
WORKSPACE=${1:-"/workspace/rn50v15_tf"}
DATA_DIR=${2:-"/data"}

GPUS_PER_NODE=8
NODE1=10.11.0.2:$GPUS_PER_NODE
NODE2=10.11.0.3:$GPUS_PER_NODE
NODE3=10.11.0.4:$GPUS_PER_NODE
NODE4=10.11.0.5:$GPUS_PER_NODE

i=1
while [ $i -le 6 ]
do
  USE_DALI=1   bash ${WORKSPACE}/resnet50v1.5/training/multi_node_train.sh ${WORKSPACE} ${DATA_DIR} \
  $GPUS_PER_NODE 120 128 fp32  $NODE1,$NODE2,$NODE3,$NODE4 $i
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  let i++
  sleep 30
done
```


# 完整日志
[ngc.zip](https://www.yuque.com/attachments/yuque/0/2020/zip/216914/1597918699364-562648c5-7e52-4e21-bd0e-bc5716f02ee4.zip?_lake_card=%7B%22uid%22%3A%221597918696451-0%22%2C%22src%22%3A%22https%3A%2F%2Fwww.yuque.com%2Fattachments%2Fyuque%2F0%2F2020%2Fzip%2F216914%2F1597918699364-562648c5-7e52-4e21-bd0e-bc5716f02ee4.zip%22%2C%22name%22%3A%22ngc.zip%22%2C%22size%22%3A9861456%2C%22type%22%3A%22application%2Fzip%22%2C%22ext%22%3A%22zip%22%2C%22progress%22%3A%7B%22percent%22%3A99%7D%2C%22status%22%3A%22done%22%2C%22percent%22%3A0%2C%22id%22%3A%22eze6V%22%2C%22card%22%3A%22file%22%7D)
# 加速比
执行以下脚本计算各个情况下的加速比：
```shell
python extract_tensorflow_logs.py --log_dir="../ngc/tensorflow"
```
输出：
```shell
../ngc/tensorflow/4n8g/r50_b128_fp32_1.log {'1': 9582.21}
../ngc/tensorflow/4n8g/r50_b128_fp32_4.log {'1': 9582.21, '4': 9548.5}
../ngc/tensorflow/4n8g/r50_b128_fp32_2.log {'1': 9582.21, '4': 9548.5, '2': 9352.08}
../ngc/tensorflow/4n8g/r50_b128_fp32_3.log {'1': 9582.21, '4': 9548.5, '2': 9352.08, '3': 9706.33}
../ngc/tensorflow/4n8g/r50_b128_fp32_6.log {'1': 9582.21, '4': 9548.5, '2': 9352.08, '3': 9706.33, '6': 9711.99}
../ngc/tensorflow/4n8g/r50_b128_fp32_5.log {'1': 9582.21, '4': 9548.5, '2': 9352.08, '3': 9706.33, '6': 9711.99, '5': 9286.44}
../ngc/tensorflow/1n8g/r50_b128_fp32_1.log {'1': 2732.05}
../ngc/tensorflow/1n8g/r50_b128_fp32_4.log {'1': 2732.05, '4': 2728.35}
../ngc/tensorflow/1n8g/r50_b128_fp32_2.log {'1': 2732.05, '4': 2728.35, '2': 2719.11}
../ngc/tensorflow/1n8g/r50_b128_fp32_3.log {'1': 2732.05, '4': 2728.35, '2': 2719.11, '3': 2720.72}
../ngc/tensorflow/1n8g/r50_b128_fp32_6.log {'1': 2732.05, '4': 2728.35, '2': 2719.11, '3': 2720.72, '6': 2722.68}
../ngc/tensorflow/1n8g/r50_b128_fp32_5.log {'1': 2732.05, '4': 2728.35, '2': 2719.11, '3': 2720.72, '6': 2722.68, '5': 2723.71}
../ngc/tensorflow/1n4g/r50_b128_fp32_1.log {'1': 1392.36}
../ngc/tensorflow/1n4g/r50_b128_fp32_4.log {'1': 1392.36, '4': 1388.01}
../ngc/tensorflow/1n4g/r50_b128_fp32_2.log {'1': 1392.36, '4': 1388.01, '2': 1390.2}
../ngc/tensorflow/1n4g/r50_b128_fp32_3.log {'1': 1392.36, '4': 1388.01, '2': 1390.2, '3': 1389.31}
../ngc/tensorflow/1n4g/r50_b128_fp32_6.log {'1': 1392.36, '4': 1388.01, '2': 1390.2, '3': 1389.31, '6': 1390.35}
../ngc/tensorflow/1n4g/r50_b128_fp32_5.log {'1': 1392.36, '4': 1388.01, '2': 1390.2, '3': 1389.31, '6': 1390.35, '5': 1393.52}
../ngc/tensorflow/1n1g/r50_b128_fp32_1.log {'1': 362.81}
../ngc/tensorflow/1n1g/r50_b128_fp32_4.log {'1': 362.81, '4': 361.67}
../ngc/tensorflow/1n1g/r50_b128_fp32_2.log {'1': 362.81, '4': 361.67, '2': 363.48}
../ngc/tensorflow/1n1g/r50_b128_fp32_3.log {'1': 362.81, '4': 361.67, '2': 363.48, '3': 363.07}
../ngc/tensorflow/1n1g/r50_b128_fp32_6.log {'1': 362.81, '4': 361.67, '2': 363.48, '3': 363.07, '6': 361.96}
../ngc/tensorflow/1n1g/r50_b128_fp32_5.log {'1': 362.81, '4': 361.67, '2': 363.48, '3': 363.07, '6': 361.96, '5': 359.55}
../ngc/tensorflow/2n8g/r50_b128_fp32_1.log {'1': 5080.61}
../ngc/tensorflow/2n8g/r50_b128_fp32_4.log {'1': 5080.61, '4': 5126.4}
../ngc/tensorflow/2n8g/r50_b128_fp32_2.log {'1': 5080.61, '4': 5126.4, '2': 5130.6}
../ngc/tensorflow/2n8g/r50_b128_fp32_3.log {'1': 5080.61, '4': 5126.4, '2': 5130.6, '3': 5105.73}
../ngc/tensorflow/2n8g/r50_b128_fp32_6.log {'1': 5080.61, '4': 5126.4, '2': 5130.6, '3': 5105.73, '6': 5089.51}
../ngc/tensorflow/2n8g/r50_b128_fp32_5.log {'1': 5080.61, '4': 5126.4, '2': 5130.6, '3': 5105.73, '6': 5089.51, '5': 5114.58}
{'r50': {'1n1g': {'average_speed': 362.09,
                  'batch_size_per_device': '128',
                  'speedup': 1.0},
         '1n4g': {'average_speed': 1390.62,
                  'batch_size_per_device': '128',
                  'speedup': 3.87},
         '1n8g': {'average_speed': 2724.44,
                  'batch_size_per_device': '128',
                  'speedup': 7.58},
         '2n8g': {'average_speed': 5107.9,
                  'batch_size_per_device': '128',
                  'speedup': 14.11},
         '4n8g': {'average_speed': 9531.26,
                  'batch_size_per_device': '128',
                  'speedup': 26.51}}}
Saving result to ./result/tensorflow_result.json
```


### ResNet50 V1.5 bsz = 128
| Node num | GPU num | OneFlow | TensorFlow | PyTorch | Paddle | NVIDIA/TF |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 383.760 | 326.96 | 347.12 | 353.66 | 362.09 |
| 1 | 2 | 747.295 | 590.94 | 604.11 | 681.4 | TODO |
| 1 | 4 | 1497.618 | 1164.24 | 899.41 | 1093.7 | 1390.62 |
| 1 | 8 | 2942.321 | 2513.72 | 1404.55 | 1302.12 | 2724.44 |
| 2 | 16 | 5839.054 | 4752.52 | 2384.3 | 5161.04 | 5107.9 |
| 4 | 32 | 11548.451 | 9407.87 | 5552.77 | 9995.53 | 9531.26, |







