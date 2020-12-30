# 【OSDI】OneFlow-Megatron-LM测评

## 概述 Overview

本测试为OSDI论文提供了多组真实测试数据，测试基于OneFlow-Benchmark仓库中的[Megatron-LM](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/gpt2_big_fc_osdi_test_1207)实现，框架依赖此分支的[oneflow](https://github.com/Oneflow-Inc/oneflow/tree/dev_optimizer_placement_optimization_nc_th)，您也可以直接pip安装我们编译好的[whl包](https://staging.oneflow.info/branch/dev_optimizer_placement_optimization_nc_th)。

基于以上版本，我们对gpt-2网络在单机单卡～4机32卡情况下进行了多组测试。



## 环境 Environment

### 系统

- #### 硬件

  - GPU：8x Tesla V100-SXM2-16GB

- #### 软件

  - 驱动：NVIDIA 440.33.01
  
  - 系统：[ Ubuntu 16.04](http://releases.ubuntu.com/16.04/)
  
  - CUDA：10.2
  
  - cuDNN：7.6.5
  
  - NCCL：2.7.3
  
  - Python：3.7.9
  
- #### 框架
  
  - **oneflow-cu102      0.3b2 ** 



## 快速开始 Quick Start

### 1.环境准备

#### 新建conda环境

推荐新建一个oneflow的conda环境

```shell
# 创建conda虚拟环境
conda create -n oneflow python=3.7.9
# 激活环境
conda activate oneflow
# 设置镜像源
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 安装oneflow
python3 -m pip install https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/dev_optimizer_placement_optimization_nc_th/930a8dc1eed2d6c11ac7fb6ceff15a2f8f312810/oneflow_cu102-0.3.2-cp37-cp37m-manylinux2014_x86_64.whl
```

#### 源码下载

```shell
# git clone OneFlow-Benchmark仓库
git clone https://github.com/Oneflow-Inc/OneFlow-Benchmark.git
cd OneFlow-Benchmark  && git checkout gpt2_big_fc_osdi_test_1207
cd LanguageModeling/gpt-2
```



### 2. 数据集准备

数据集使用Wikipedia数据集，并使用[wikiextractor](https://github.com/attardi/wikiextractor)将原始文件解析成txt格式的文本文件。

#### 下载wiki数据集

```
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```
#### 数据集解析
```python
python -m wikiextractor.WikiExtractor /your_path_to/enwiki-latest-pages-articles.xml.bz2  --json --no_templates -o /datasets/wiki/enwiki --processes 48
```

### 3.测试

本次测试集群中有 4 台节点：

- NODE1=10.11.0.2
- NODE2=10.11.0.3
- NODE3=10.11.0.4
- NODE4=10.11.0.5

每个节点有 8 张 V100 显卡， 每张显卡显存 16 GB。



#### 参数及配置

测试使用[train_gpt2.sh](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/gpt2_big_fc_osdi_test_1207/LanguageModeling/gpt-2/src/train_gpt2.sh)脚本，其中，默认使用gpt-2 small的网络配置，主要参数如下：

- batch_size_per_device为单卡batch_size，默认为4

- gpus 每台机器使用的gpu，默认为从0～7号卡（8张gpu）

- dtype 数据类型，默认为fp16混合精度

- non_distributed_optimizer 是否开启Optimizer-Placement Optimization优化，默认为off关闭;on可打开

- num_node 分布式训练集群中，机器节点数（单机情况下设为1；2机情况下设为2，以此类推）

- node_ips 训练集群的ip字符串


**默认测试的网络为gpt2-small** ，也可以修改脚本中的参数设置不同型号的gpt2网络，如：

```shell
# gpt2-small
num_layers=12
num_attention_heads=12
hidden_size=768
# # gpt2-medium
# num_layers=24
# num_attention_heads=16
# hidden_size=1024
```

#### 单机测试

运行以下脚本将对单机单卡～8卡进行测试

```shell
# 单机1卡（注：off表示关闭Optimizer-Placement Optimization优化；4表示测试batch size=4）
bash src/train_gpt2.sh off  0  4
# 单机4卡
bash src/train_gpt2.sh off  0,1,2,3  4
# 单机8卡
bash src/train_gpt2.sh off  0,1,2,3,4,5,6,7  4
```

#### 分布式测试

修改train_gpt2.sh中位于[22~23](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/gpt2_big_fc_osdi_test_1207/LanguageModeling/gpt-2/src/train_gpt2.sh#L22)行的num_node、node_ips变量

二机：

```shell
num_node=2
node_ips="10.11.0.2,10.11.0.3"
```

然后执行`bash src/train_gpt2.sh off  0,1,2,3,4,5,6,7  4`

四机：

```shell
num_node=4
node_ips="10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"
```

然后执行`bash src/train_gpt2.sh off  0,1,2,3,4,5,6,7  4`



## 测试结果 Performance

#### 符号说明

- xn_xg_xdp_xmp_xbs表示x node, x gpu, x data parallel, x model parallel, x batch size per gpu

**单机8卡**

| test_num | test_desc | xn_xg_xdp_xmp_xbs | gpu_mem(mB)  | gpu_util(%)        | throuthput(sample/sec) |            |
| -------- | --------- | ----------------- | ------------ | ------------------ | ---------------------- | ---------- |
| 20201208 | test-10-1 | off_optimization  | 1n_8g_dp_1bs | 5603               | 73                     | 104        |
|          | test-10-2 | on_optimization   | 1n_8g_dp_1bs | 3849(-1754,↓31.3%) | 43                     | 89(↓14.4%) |
|          | test-10-3 | off_optimization  | 1n_8g_dp_4bs | 11301              | 95                     | 239        |
|          | test-10-4 | on_optimization   | 1n_8g_dp_4bs | 9547(-1754,↓15.5%) | 88                     | 235(↓1.7%) |

**2机16卡**

| date     | test_num  | test_desc        | xn_xg_xdp_xmp_xbs | gpu_mem(mB)        | gpu_util(%) | throuthput(sample/sec) |
| -------- | --------- | ---------------- | ----------------- | ------------------ | ----------- | ---------------------- |
| 20201208 | test-14-1 | off_optimization | 2n_8g_dp_4bs      | 11087              | 96          | 412                    |
|          | test-14-2 | on_optimization  | 2n_8g_dp_4bs      | 9160(-1927,11.4↓%) | 92          | 395(4.1↓%)             |

**4机32卡**

| date     | test_num  | test_desc        | xn_xg_xdp_xmp_xbs | gpu_mem(mB)        | gpu_util(%) | throuthput(sample/sec) |
| -------- | --------- | ---------------- | ----------------- | ------------------ | ----------- | ---------------------- |
| 20201209 | test-19-1 | off_optimization | 4n_8g_dp_4bs      | 11089              | 95          | 717                    |
|          | test-19-2 | on_optimization  | 4n_8g_dp_4bs      | 9082(-2727,↓23.1%) | 92          | 742(↑3.4%)             |

### 日志下载

详细 Log 信息可点击下载：[osdi-oneflow-gpt2-logs-vs-zero.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/Megatron-LM/osdi-oneflow-gpt2-logs-vs-zero.zip)


