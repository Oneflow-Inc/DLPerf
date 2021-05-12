# [NVIDIA](https://github.com/NVIDIA)/[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)复现

## 概述
  本次测评提供了多组基于真实数据集的测试结果。测评基于[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).commit为`8aa4619f2b2a57b5725026a50ebd2b15e8121482`.基于以上，对Megatron-LM进行了从单卡到多机、从DP、MP、2D到PP等等的测试。
- ### 测评背景
  2020年OpenAI发布了GPT-3。于是在大规模模型的今天，训练超大规模模型所需的算力、存储已经不再是单机或几台机器能搞定的了，因此，对于如何解决分布式训练、如何能让大众也有能力搭建大规模模型都有着极高的挑战。基于此，NVIDIA提出了[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)，基于PyTorch进行分布式训练GPT。
- ### 测评目的
  测试对比Megatron-LM在各种情况下的内存使用率、速度等

## 环境
### 系统
- ### 硬件
  - GPU：8x Tesla V100-SXM2-16GB
- ### 软件
  - NCCL version 2.8.3+cuda11.1
  - 容器：[NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) version 20.12，`docker pull nvcr.io/nvidia/pytorch:20.12-py3`
  - Megatron-LM的commit为：[`8aa4619f2b2a57b5725026a50ebd2b15e8121482`](https://github.com/NVIDIA/Megatron-LM/commit/8aa4619f2b2a57b5725026a50ebd2b15e8121482)
  - 其他环境信息参考[这里](https://github.com/NVIDIA/Megatron-LM#setup)

## 快速开始
- ### docker准备
  - 根据官网[要求](https://github.com/NVIDIA/Megatron-LM#setup)，获取NGC's PyTorch container容器，`docker pull nvcr.io/nvidia/pytorch:20.12-py3`。类脑环境下存在镜像，可直接`docker load -i /data/docker_images/ngc-pytorch-20.12-py3.tar`. 检查是否成功：`docker images | grep ad0f29ddeb63`
  - 进入docker容器，进行配置，如：`docker run --rm -it -v --shm-size=16g --ulimit memlock=-1 --privileged --name gpt_megatron_lm --net host -v /data/Megatron-LM:/data/Megatron-LM -v /data/gpt:/data/gpt nvcr.io/nvidia/pytorch:20.12-py3`
  - 安装IB驱动
    - 宿主机上需要有IB驱动，先查看所有宿主机上的驱动版本,`ofed_info -s`,并保持一致。docker里的驱动版本一定要小于宿主机的版本，否则会导致不能使用到IB网络而只能使用Socket通信。
    - 查看docker里系统版本`cat /etc/os-release`，从`https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed`选择对应的版本，如[MLNX_OFED_LINUX-5.3-1.0.0.1-ubuntu20.04-x86_64.tgz](https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.0.1/MLNX_OFED_LINUX-5.3-1.0.0.1-ubuntu20.04-x86_64.tgz)
    - 更换阿里云源，`cp /etc/apt/sources.list /etc/apt/sources.list.bak && vim /etc/apt/sources.list`,将下列源址复制进`/etc/apt/sources.list`中
      
      ```
      deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
      deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
      deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
      deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
      deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
      deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
      deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
      deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
      deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
      deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse

      ```

    - 安装依赖
      
      ```
      apt-get update
      apt install dpatch libelf1 libmnl0 libltdl-dev lsof chrpath debhelper pciutils tk bison graphviz ethtool kmod gfortran swig flex tcl
      apt-get install -y libusb-1.0-0 udev libfuse2 libgfortran4
      ```
    
    - 安装
      ```
      mkdir tmp && tar zxvf MLNX_OFED_LINUX-5.3-1.0.0.1-ubuntu20.04-x86_64.tgz -C ./tmp/ && cd tmp/MLNX_OFED_LINUX-5.3-1.0.0.1-ubuntu20.04-x86_64
      ./mlnxofedinstall --user-space-only --without-fw-update --all --force 
      ```
    - 测试`ibstat`,输出如下即表示成功
      ```
      CA 'mlx5_0'
              CA type: MT4115
              Number of ports: 1
              Firmware version: 12.28.2006
              Hardware version: 0
              Node GUID: 0x506b4b0300f371dc
              System image GUID: 0x506b4b0300f371dc
              Port 1:
                State: Active
                Physical state: LinkUp
                Rate: 100
                Base lid: 99
                LMC: 0
                SM lid: 27
                Capability mask: 0x2651e848
                Port GUID: 0x506b4b0300f371dc
                Link layer: InfiniBand
      ```
    - 配置Docker间免密登录



- ### 数据集

- ####  OpenWebText
  根据官方[说明](https://github.com/NVIDIA/Megatron-LM#datasets)，从[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)下载数据集.
  - 其中，`openwebtext.tar.xz`为原始数据集、`openwebtext`为`json`文件。`gpt_sample_dataset_text_document.bin`和`gpt_sample_dataset_text_document.idx`为最终生成的数据集文件
- 数据集制作过程
  - `tar -xvJf openwebtext.tar.xz -C /datasets`，得到的是形如`urlsf_subset20-93_data.xz`的文件，再解压`xz -d ./*`，使用下面`openweb_to_json.py`转换为json格式的文件。

    ```python
    import sys
    import tarfile
    import json
    with tarfile.open(sys.argv[1], 'r') as tar:
      for member in tar.getmembers():
        print(json.dumps({'url': member.name, 'text': str(tar.extractfile(member).read())}))
        
    ```

  - `ls | xargs -n 1 -P 96 -I {} sh -c 'python openweb_to_json.py {} > ../openwebtext-json/{}.json'`即可得到数据集

  - 根据官方[Data Preprocessing](https://github.com/NVIDIA/Megatron-LM#data-preprocessing)说明，制作数据集，运行下面shell脚本，`bash create_dataset.sh`得到数据集文件`gpt_sample_dataset_text_document.bin`和`gpt_sample_dataset_text_document.idx`

- ### 脚本与配置
  - `git clone https://github.com/NVIDIA/Megatron-LM.git`下载源码,`git checkout 8aa4619f2b2a57b5725026a50ebd2b15e8121482`
  - 下载`gpt2-vocab.json`和`gpt2-merges.txt`文件：`wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json`、`wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt`
  - 加载镜像，并运行，进入容器，如`docker attach 31e`

## 测试
  本次测试集群中有 4 台节点：
  - NODE1=10.11.0.2
  - NODE2=10.11.0.3
  - NODE3=10.11.0.4
  - NODE4=10.11.0.5
  每个节点有 8 张 V100 显卡， 每张显卡显存 16 GB。
- ### 参数及配置
  测试使用的脚本在`scripts`目录下，可将其移动到Megatron-LM/examples下。其中确定了`tensor-model-parallel-size`、`pipeline-model-parallel-size`、`NNODES`以及`GPUS_PER_NODE`等参数情况下，即可确定`data_parallel_size`，即`DP*MP*PP=NODES*GPUS_PER_NODE`
  - M_P=${1:-1}，`tensor-model-parallel-size`，指定了模型张量呗切分到多少个GPU设备上，即`张量模型并行度`
  - P_P=${2:-1}，`pipeline-model-parallel-size`，即`流水模型并行度`，如一个24层的网络，如果`pipeline-model-parallel-size=4`，即表示将24层分为4个stage，每个stage都会由一组GPU设备处理
  - MICRO_BATCH_SIZE=${3:-8}，是每组模型并行的设备使用的`batch size`，如果是纯数据并行，就是每卡`batch size`
  - GLOABAL_BATCH_SIZE=${4:-16}，`micro_batch_size * data_parallel_size = micro_batch_times_data_parallel_size`
  - GPUS_PER_NODE=${8:-8}，每个节点的GPU数量
  - NNODES=${5:-1}，节点数量
  - MASTER_ADDR=${6:-10.11.0.2}，master节点ip地址
  - MASTER_PORT=21327，通信端口
  - NODE_RANK=${7:-0}，当前节点编号
  - TRAIN_ITERS=${9:-520}，训练的轮数
- ### 运行脚本示例
  ```
    #单机单卡
    bash examples/mutil_perf_pretrain_gpt_dp_mp_pp_1536_16_16.sh 1 1 2 2 1 10.11.0.2 0 1
    #4机32卡情况下，分别执行
    bash examples/mutil_perf_pretrain_gpt_dp_mp_pp.sh 8 1 8 512 4 10.11.0.2 0
  ```

## 测试结果
- ### 测试环境
  所有的测试都是在4台配置了8张 V100-SXM2-16GB GPU的服务器中，主要硬软件配置信息：
  ```
    Tesla V100-SXM2-16GB x 8
    InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
    Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
    Memory 384G
    Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
    CUDA Version: 10.2, Driver Version: 440.33.01

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

- ### 测试用例
group | case | num-nodes | num-gpus-per-node | data-parallel-size | tensor-model-parallel-size | pipeline-model-parallel-size | micro-batch-size | micro-batch-size-times-data-parallel-size | num-accumulation-steps | global-batch-size | hidden-size | num-attention-heads | num-layers
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
DP |  |  |  |  |  |  |  |  |  |  |  |  | 
 -- | DP_1x1x1_2_1536x16 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 1536 | 16 | 16
 -- | DP_8x1x1_16_1536x16 | 1 | 8 | 8 | 1 | 1 | 2 | 16 | 1 | 16 | 1536 | 16 | 16
 -- | DP_16x1x1_32_1536x16 | 2 | 8 | 16 | 1 | 1 | 2 | 32 | 1 | 32 | 1536 | 16 | 16
 -- | DP_32x1x1_64_1536x16 | 4 | 8 | 32 | 1 | 1 | 2 | 64 | 1 | 64 | 1536 | 16 | 16
MP |  |  |  |  |  |  |  |  |  |  |  |  | 
 -- | MP_1x1x1_8_768x12 | 1 | 1 | 1 | 1 | 1 | 8 | 8 | 1 | 8 | 768 | 12 | 12
 -- | MP_1x8x1_16_1536x16 | 1 | 8 | 1 | 8 | 1 | 16 | 16 | 1 | 16 | 1536 | 16 | 16
 -- | MP_1x16x1_16_3072x16 | 2 | 8 | 1 | 16 | 1 | 16 | 16 | 1 | 16 | 3072 | 32 | 16
 -- | MP_1x32x1_16_3072x32 | 4 | 8 | 1 | 32 | 1 | 16 | 16 | 1 | 16 | 3072 | 32 | 32
2D |  |  |  |  |  |  |  |  |  |  |  |  | 
 -- | 2D_8x1x1_16_1536x16 | 1 | 8 | 8 | 1 | 1 | 2 | 16 | 1 | 16 | 1536 | 16 | 16
 -- | 2D_1x8x1_16_1536x16 | 1 | 8 | 1 | 8 | 1 | 16 | 16 | 1 | 16 | 1536 | 16 | 16
 -- | 2D_2x4x1_16_1536x16 | 1 | 8 | 2 | 4 | 1 | 8 | 16 | 1 | 16 | 1536 | 16 | 16
 -- | 2D_4x2x1_16_1536x16 | 1 | 8 | 4 | 2 | 1 | 4 | 16 | 1 | 16 | 1536 | 16 | 16
 -- | 2D_2x8x1_16_2304x24 | 2 | 8 | 2 | 8 | 1 | 8 | 16 | 1 | 16 | 2304 | 24 | 24
 -- | 2D_4x8x1_32_2304x24 | 4 | 8 | 4 | 8 | 1 | 8 | 32 | 1 | 32 | 2304 | 24 | 24
PP |  |  |  |  |  |  |  |  |  |  |  |  | 
 -- | DP_PP_8x1x4_512_1536x16 | 4 | 8 | 8 | 1 | 4 | 2 | 16 | 32 | 512 | 1536 | 16 | 16
 -- | MP_PP_1x8x4_512_2304x24 | 4 | 8 | 1 | 8 | 4 | 16 | 16 | 32 | 512 | 2304 | 24 | 24
 -- | 2D_PP_2x4x4_512_2304x24 | 4 | 8 | 2 | 4 | 4 | 8 | 16 | 32 | 512 | 2304 | 24 | 24
 -- | 2D_PP_2x8x2_512_2304x24 | 4 | 8 | 2 | 8 | 2 | 8 | 16 | 32 | 512 | 2304 | 24 | 24
 -- |  |  |  |  |  |  |  |  |  |  |  |  | 

- ### 测试日志
  所有日志都在`https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/`下。  
  
group | case | megatron_logs
 -- | -- | --
DP |  | 
--  | DP_1x1x1_2_1536x16 | perf_test_logs/megatron_lm_perf_1n1g_dp1_mp1_pp1_mbs2_gbs2_pretrain_0.log
--  | DP_8x1x1_16_1536x16 | perf_test_logs/megatron_lm_perf_1n8g_dp8_mp1_pp1_mbs2_gbs16_pretrain_0.log
--  | DP_16x1x1_32_1536x16 | perf_test_logs/megatron_lm_perf_2n8g_dp16_mp1_pp1_mbs2_gbs32_pretrain_[0-1].log
--  | DP_32x1x1_64_1536x16 | perf_test_logs/megatron_lm_perf_4n8g_dp32_mp1_pp1_mbs2_gbs64_pretrain_[0-3].log
MP |  | 
--  | MP_1x1x1_8_768x12 | megatron_perf_logs/megatron_pretrain_gpt_1n1d_dp1_mp1_pp1_mbz8_gbz8_s2048_l12_h768_nh12_rank0.log
--  | MP_1x8x1_16_1536x16 | perf_test_logs/megatron_lm_perf_1n8g_dp1_mp8_pp1_mbs16_gbs16_pretrain_0.log
--  | MP_1x16x1_16_3072x16 | perf_test_logs/megatron_lm_perf_2n8g_dp1_mp16_pp1_mbs16_gbs16_MP_1x16x1_16_3072x16_[0-1].log
--  | MP_1x32x1_16_3072x32 | perf_test_logs/megatron_lm_perf_4n8g_dp1_mp32_pp1_mbs16_gbs16_pretrain_[0-3].log
2D |  | 
 -- | 2D_8x1x1_16_1536x16 | megatron_perf_logs/megatron_pretrain_gpt_1n8d_dp8_mp1_pp1_mbz2_gbz16_s2048_l16_h1536_nh16_rank0.log
 -- | 2D_1x8x1_16_1536x16 | megatron_perf_logs/megatron_pretrain_gpt_1n8d_dp1_mp8_pp1_mbz16_gbz16_s2048_l16_h1536_nh16_rank0.log
 -- | 2D_2x4x1_16_1536x16 | megatron_perf_logs/megatron_pretrain_gpt_1n8d_dp2_mp4_pp1_mbz8_gbz16_s2048_l16_h1536_nh16_rank0.log
 -- | 2D_4x2x1_16_1536x16 | megatron_perf_logs/megatron_pretrain_gpt_1n8d_dp4_mp2_pp1_mbz4_gbz16_s2048_l16_h1536_nh16_rank0.log
 -- | 2D_2x8x1_16_2304x24 | megatron_perf_logs/megatron_pretrain_gpt_2n8d_dp2_mp8_pp1_mbz8_gbz16_s2048_l24_h2304_nh24_rank[0-1].log
 -- | 2D_4x8x1_32_2304x24 | megatron_perf_logs/megatron_pretrain_gpt_4n8d_dp4_mp8_pp1_mbz8_gbz32_s2048_l24_h2304_nh24_rank[0-3].log
PP |  | 
 -- | DP_PP_8x1x4_512_1536x16 | oneflow_perf_logs/megatron_lm_perf_4n8g_dp8_mp1_pp4_mbs2_gbs512_pretrain_1536_16_16_[0-3].log
 -- | MP_PP_1x8x4_512_2304x24 | oneflow_perf_logs/megatron_lm_perf_4n8g_dp1_mp8_pp4_mbs16_gbs512_pretrain_[0-3].log
 -- | 2D_PP_2x4x4_512_2304x24 | oneflow_perf_logs/megatron_pretrain_gpt_4n8d_dp2_mp4_pp4_mbz8_gbz512_s2048_l24_h2304_nh24_rank[0-3].log; oneflow_perf_logs/megatron_lm_perf_4n8g_dp2_mp4_pp4_mbs8_gbs512_pretrain_[0-3].log
 -- | 2D_PP_2x8x2_512_2304x24 | oneflow_perf_logs/megatron_pretrain_gpt_4n8d_dp2_mp8_pp2_mbz8_gbz512_s2048_l24_h2304_nh24_rank[0-3].log; oneflow_perf_logs/megatron_lm_perf_4n8g_dp2_mp8_pp2_mbs8_gbs512_pretrain_[0-3].log

- ### 测试结果
group  |  case  |  lantency-  |  memory-
--  |  --  |  --  |  --
DP  |    |    |  
 --  |  DP_1x1x1_2_1536x16  |  464.12 (ms)  |  14362 (MiB)
 --  |  DP_8x1x1_16_1536x16  |  480.96 (ms)  |  14938 (MiB)
 --  |  DP_16x1x1_32_1536x16  |  664.46 (ms)  |  14508 (MiB)
 --  |  DP_32x1x1_64_1536x16  |  683.51 (ms)  |  14508 (MiB)
MP  |    |    |  
 --  |  MP_1x1x1_8_768x12  |  611.2 (ms)  |  9810 (MiB)
 --  |  MP_1x8x1_16_1536x16  |  692.44 (ms)  |  12684 (MiB)
 --  |  MP_1x16x1_16_3072x16  |  5610.23(ms)  |  12150 (MiB)
 --  |  MP_1x32x1_16_3072x32  |  13037.85 (ms)  |  8834 (MiB)
2D  |    |    |  
 --  |  2D_8x1x1_16_1536x16  |  480.22 (ms)  |  14938 (MiB)
 --  |  2D_1x8x1_16_1536x16  |  664.64 (ms)  |  12684 (MiB)
 --  |  2D_2x4x1_16_1536x16  |  576.48 (ms)  |  9104 (MiB)
 --  |  2D_4x2x1_16_1536x16  |  593.68 (ms)  |  9642 (MiB)
 --  |  2D_2x8x1_16_2304x24  |  1313.52 (ms)  |  14928 (MiB)
 --  |  2D_4x8x1_32_2304x24  |  1508.48 (ms)  |  15004 (MiB)
PP  |    |    |  
 --  |  DP_PP_8x1x4_512_1536x16  |  4893.55 (ms)  |  7130 (MiB)
 --  |  MP_PP_1x8x4_512_2304x24  |  22166.85 (ms)  |  14204 (MiB)
 --  |  2D_PP_2x4x4_512_2304x24  |  15331.23 (ms)  |  10018 (MiB)
 --  |  2D_PP_2x8x2_512_2304x24  |  16281.51 (ms)  |  10554 (MiB)



