# OneFlow GPT Pretrain Performance Benchmark

本目录包含了 OneFlow GPT 预训练性能测试的环境、脚本以及结果。
  
## 测试环境

实现 OneFlow GPT 相关的源码仓库如下：

- OneFlow repo and commit: [OneFlow#75f11b8](https://github.com/Oneflow-Inc/oneflow/commit/75f11b8257112c7afd0c777abf7cddc01b6b495c)
- OneFlow-Benchmark repo and commit: [OneFlow-Benchmark#8da0250](https://github.com/Oneflow-Inc/OneFlow-Benchmark/commit/8da0250ad407da338825f1eb3f15ab8c7461427d)

OneFlow GPT Pretrain 相关的模型脚本、启动脚本和工具全部在 OneFlow-Benchmark repo 的 `LanguageModeling/GPT/` 目录下。

我们使用的测试资源是 4 台各配置了 8 张 V100-SXM2-16GB GPU 的服务器中进行。主要的硬软件配置如下：

- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
- Python 3.7

单台服务器内 8 张 GPU 卡拓扑结构使用 `nvidia-smi topo -m` 命令得知如下：

```
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

## 测试用例

| group | case | num_nodes | num_gpus_per_node | data_parallel_size | tensor_model_parallel_size | pipeline_model_parallel_size | micro_batch_size | micro_batch_size_times_data_parallel_size | num_accumulation_steps | global_batch_size | hidden_size | num_attention_heads | num_layers
| - | - | - | - | - | - | - | - | - | - | - | - | - | - 
| DP |  |  |  |  |  |  |  |  |  |  |  |  | 
| | DP_1x1x1_2_1536x16 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 1536 | 16 | 16
| | DP_8x1x1_16_1536x16 | 1 | 8 | 8 | 1 | 1 | 2 | 16 | 1 | 16 | 1536 | 16 | 16
| | DP_16x1x1_32_1536x16 | 2 | 8 | 16 | 1 | 1 | 2 | 32 | 1 | 32 | 1536 | 16 | 16
| | DP_32x1x1_64_1536x16 | 4 | 8 | 32 | 1 | 1 | 2 | 64 | 1 | 64 | 1536 | 16 | 16
| MP |  |  |  |  |  |  |  |  |  |  |  |  | 
| | MP_1x1x1_8_768x12 | 1 | 1 | 1 | 1 | 1 | 8 | 8 | 1 | 8 | 768 | 12 | 12
| | MP_1x8x1_16_1536x16 | 1 | 8 | 1 | 8 | 1 | 16 | 16 | 1 | 16 | 1536 | 16 | 16
| | MP_1x16x1_16_3072x16 | 2 | 8 | 1 | 16 | 1 | 16 | 16 | 1 | 16 | 3072 | 32 | 16
| | MP_1x32x1_16_3072x32 | 4 | 8 | 1 | 32 | 1 | 16 | 16 | 1 | 16 | 3072 | 32 | 32
| 2D |  |  |  |  |  |  |  |  |  |  |  |  | 
| | 2D_8x1x1_16_1536x16 | 1 | 8 | 8 | 1 | 1 | 2 | 16 | 1 | 16 | 1536 | 16 | 16
| | 2D_1x8x1_16_1536x16 | 1 | 8 | 1 | 8 | 1 | 16 | 16 | 1 | 16 | 1536 | 16 | 16
| | 2D_2x4x1_16_1536x16 | 1 | 8 | 2 | 4 | 1 | 8 | 16 | 1 | 16 | 1536 | 16 | 16
| | 2D_4x2x1_16_1536x16 | 1 | 8 | 4 | 2 | 1 | 4 | 16 | 1 | 16 | 1536 | 16 | 16
| | 2D_2x8x1_16_2304x24 | 2 | 8 | 2 | 8 | 1 | 8 | 16 | 1 | 16 | 2304 | 24 | 24
| | 2D_4x8x1_32_2304x24 | 4 | 8 | 4 | 8 | 1 | 8 | 32 | 1 | 32 | 2304 | 24 | 24
| PP |  |  |  |  |  |  |  |  |  |  |  |  | 
| | DP_PP_8x1x4_512_1536x16 | 4 | 8 | 8 | 1 | 4 | 2 | 16 | 32 | 512 | 1536 | 16 | 16
| | MP_PP_1x8x4_512_2304x24 | 4 | 8 | 1 | 8 | 4 | 16 | 16 | 32 | 512 | 2304 | 24 | 24
| | 2D_PP_2x4x4_512_2304x24 | 4 | 8 | 2 | 4 | 4 | 8 | 16 | 32 | 512 | 2304 | 24 | 24
| | 2D_PP_2x8x2_512_2304x24 | 4 | 8 | 2 | 8 | 2 | 8 | 16 | 32 | 512 | 2304 | 24 | 24
| |  |  |  |  |  |  |  |  |  |  |  |  | 

group 中的 DP, MP, 2D, PP 分别代表了**数据并行（data parallel）**, **模型并行（tensor model parallel）**, **数据模型混合并行（2-D parallel）**, **流水并行（pipeline model parallel）**。

case 是根据不同的并行方式和不同的 batch size 和模型大小来选取的。比如 `4x8x1` 代表着并行方式，其中数字分别代表 `data_parallel size=4, tensor_model_parallel_size=8, pipeline_model_parallel_size=1`。最后的 `1536x16` 代表着模型大小，其中数字分别代表 `hidden_size=1536, num_layers=16`。中间的单独的数字 `512` 代表着 `global_batch_size`。这些参数的不同组合组成了不同的 case。同时 case 也会按照一定规律来组合，比如 DP 分组中的所有 case 在模型大小不变的情况下 batch size 跟着参与计算的设备数增加而线性增加。MP 分组中的 case 则是在 batch size 不变的情况下，根据并行的设备数增加而增加模型的大小。2D 分组中分别测试了在相同硬件资源下，不同并行方式的 case 和增加硬件资源下的 case。而 PP 分组包含着**梯度累加**参数（num_accumulation_steps>1）。

同时注意设置这些参数需要满足以下关系，不是随意设置的

- `micro_batch_size * data_parallel_size * num_accumulation_steps == global_batch_size`
- `data_parallel_size * tensor_model_parallel_size * pipeline_model_parallel_size == num_nodes * num_gpus_per_node`
- `hidden_size % num_attention_heads == 0`
- `num_attention_heads % tensor_model_parallel_size == 0`
- `num_layers % pipeline_model_parallel_size == 0`

## 测试脚本

### 环境配置

在测试之前需要先安装 oneflow package 和 oneflow_gpt package。

oneflow package 由上文提到的 [OneFlow#75f11b8](https://github.com/Oneflow-Inc/oneflow/commit/75f11b8257112c7afd0c777abf7cddc01b6b495c) 版本编译打包而成。这里 [oneflow-0.3.5+cu112.git.75f11b825-cp37-cp37m-manylinux2014_x86_64.whl](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/package/oneflow-0.3.5%2Bcu112.git.75f11b825-cp37-cp37m-manylinux2014_x86_64.whl) 提供直接下载链接。下完成后可以通过 pip 命令安装：

```shell
python3 -m pip install oneflow-0.3.5+cu112.git.75f11b825-cp37-cp37m-manylinux2014_x86_64.whl
```

oneflow_gpt package 由上文提到的 [OneFlow-Benchmark#8da0250](https://github.com/Oneflow-Inc/OneFlow-Benchmark/commit/8da0250ad407da338825f1eb3f15ab8c7461427d) 中的 `LanguageModeling/GPT/` 目录 pip install 而来：

```shell
git clone https://github.com/Oneflow-Inc/OneFlow-Benchmark.git
cd OneFlow-Benchmark/LanguageModeling/GPT
git checkout 8da0250
python3 -m pip install -e .
```

### 数据集准备

从 [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/) 下载数据集。下载完成后得到一个压缩文件 `openwebtext.tar.xz`。使用脚本 `scripts/openweb_to_json.py` 转化得到 json 格式的语料文件。下载[词表](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json)和[分词](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt)文件，再借用 Megatron-LM 的 [preprocess_data.py](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py) 脚本来产生最后的文档二进制文件（如 `gpt_sample_dataset_text_document.bin`）和索引文件（如 `gpt_sample_dataset_text_document.idx`）。OneFlow GPT 可以直接读取这两个文件来进行训练。具体操作过程如下：

```
tar -xvJf openwebtext.tar.xz
ls openwebtext  
# 可以看到 openwebtext 目录下都是形如 urlsf_subset20-93_data.xz 的压缩文件，需要进一步解压
xz -d openwebtext/*

# 使用 openweb_to_json.py 将原始文本文件转成 json 格式文件
mkdir openwebtext-json
ls openwebtext | xargs -n 1 -P 48 -I{} bash -c 'python3 openweb_to_json.py {} > openwebtext-json/{}.json'
cat openwebtext-json/*.json > openwebtext.json

# 下载词表
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json

# 下载 BPE 分词所需的 merge 文件
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

# 使用 preprocess_data.py 根据 json 文件生成二进制文档和索引文件
python /path/to/Megatron-LM/tools/preprocess_data.py \
    --input openwebtext.json \
    --output-prefix gpt_sample_dataset \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod

# 输出文件为 gpt_sample_dataset_text_document.bin 和 gpt_sample_dataset_text_document.dix，然后可以将 /path/to/gpt_sample_dataset_text_document 直接作为 --dataset 参数传入到 oneflow_gpt.tranining 中去
```

### 测试脚本参数配置

测试脚本位置 `scripts/pretrain.sh`。

需要根据上文测试用例中各 case 的参数配置来修改脚本中对应的变量的值。其中 `data_parallel_size` 和 `num_accumulation_steps` 不能直接配置，但它们可以根据上文提到的参数关系由其他参数推导而来。例如: 

```shell
num_node=1
num_gpus_per_node=8
tensor_model_parallel_size=4
# 可以推导出: data_parallel_size=2 (num_node * num_gpus_per_node / tensor_model_parallel_size)

micro_batch_size=8
global_batch_size=512
# 可以推导出: num_accumulation_steps=32 (global_batch_size / micro_batch_size / data_parallel_size)
```

其他可以配置的参数：

- `dataset`: 需要根据你的 gpt_sample_dataset_text_document.bin 和 gpt_sample_dataset_text_document.idx 路径来配置，参考 OneFlow GPT [数据预处理](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/complete_gpt/LanguageModeling/GPT#gpt-%E9%A2%84%E8%AE%AD%E7%BB%83)一节。
- `node_ips`: 需要根据你实际测试机器所能访问到的 ip 地址来配置。
- `seq_length`: 测试用例中的所有 case 都被固定为 2048，也可根据需求自行调整（比如改为 1024）。
- `train_iters` 和 `log_interval`: 测试所跑的总轮次和每多少轮打印一次统计。

脚本中关于学习率、优化器参数、词表大小、数据集分块、模型保存和加载等等其他参数用户也可直接修改。oneflow_gpt.training 支持的完整参数列表见 [config.py](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/LanguageModeling/GPT/oneflow_gpt/config.py)。

脚本配置完毕后即可直接执行脚本来完成测试：

```
bash scripts/pretrain.sh
```

其打印输出如下:

```
 > padded vocab (size: 50257) with 431 dummy tokens (new size: 50688)
------------------------ arguments ------------------------
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  apply_query_key_layer_scaling ................... True
  attention_dropout ............................... 0.1
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  checkpoint_activations .......................... True
  clip_grad ....................................... 1.0
  ctrl_port ....................................... 50051
  data_parallel_size .............................. 2
  dataset ......................................... /data/gpt/gpt_sample_dataset_text_document
  fp16 ............................................ True
  global_batch_size ............................... 512
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 2304
  init_method_std ................................. 0.02
  initial_loss_scale .............................. 1048576
  load ............................................ None
  log ............................................. ./output
  log_interval .................................... 10
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 320000
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 3200
  make_vocab_size_divisible_by .................... 128
  metric_print_format ............................. table
  micro_batch_size ................................ 8
  min_lr .......................................... 1e-05
  multihead_attention_fusion ...................... True
  node_ips ........................................ ['10.11.0.2', '10.11.0.3', '10.11.0.4', '10.11.0.5']
  num_accumulation_steps .......................... 32
  num_attention_heads ............................. 24
  num_gpus_per_node ............................... 8
  num_layers ...................................... 24
  num_nodes ....................................... 4
  optimizer ....................................... adamw
  padded_vocab_size ............................... 50688
  pipeline_model_parallel_size .................... 4
  profile_transformer_layer ....................... False
  save ............................................ model_save
  save_interval ................................... 10000
  save_last ....................................... False
  scale_tril_softmax_dropout_fusion ............... True
  seed ............................................ 12345
  seq_length ...................................... 2048
  split ........................................... [949, 50, 1]
  tensor_model_parallel_size ...................... 4
  train_iters ..................................... 100
  train_samples ................................... 51200
  use_external_dataset ............................ False
  use_rdma ........................................ True
  vocab_size ...................................... 50257
  weight_decay .................................... 0.01
-------------------- end of arguments ---------------------
Training...
utilization.gpu [%], memory.used [MiB]
100 %, 11838 MiB
81 %, 11838 MiB
79 %, 11958 MiB
81 %, 11958 MiB
86 %, 11838 MiB
79 %, 11838 MiB
80 %, 11958 MiB
100 %, 11958 MiB
| batches  | micro_batches   | samples         | throughput | latency    | loss       |
| -------- | --------------- | --------------- | ---------- | ---------- | ---------- |
| 10       | 320             | 5120            | 35.88042   | 14.26962   | 11.23708   |
| 20       | 640             | 10240           | 39.02016   | 13.12142   | 10.43741   |
| 30       | 960             | 15360           | 39.01045   | 13.12469   | 9.02948    |
| 40       | 1280            | 20480           | 39.01279   | 13.12390   | 8.20573    |
| 50       | 1600            | 25600           | 38.99669   | 13.12932   | 7.68777    |
| 60       | 1920            | 30720           | 39.00954   | 13.12500   | 7.42809    |
| 70       | 2240            | 35840           | 39.04432   | 13.11330   | 7.22044    |
| 80       | 2560            | 40960           | 39.05294   | 13.11041   | 7.08075    |
| 90       | 2880            | 46080           | 39.04735   | 13.11229   | 6.95465    |
| 100      | 3200            | 51200           | 39.00969   | 13.12494   | 6.82232    |
```

分别打印了该次 pretrain 运行的参数配置，内存使用情况和按轮次的统计数据。

### 多机启动

多机环境下需要在每台机器上配置相同环境，并且同时运行 `pretrain.sh` 即可。

### 在容器中启动

在容器环境下运行测试可以帮助节省环境配置的时间。我们使用的容器镜像为 oneflow-manylinux2014-cuda11.2:0.1，可以根据 OneFlow 提供的 [dockerfile](https://github.com/Oneflow-Inc/oneflow/tree/master/docker/package/manylinux/Dockerfile) 来 build，或者直接[下载](https://oneflow-static.oss-cn-beijing.aliyuncs.com/docker_images/oneflow-manylinux2014-cuda11.2-0.1.tar.gz)。镜像 build 成功或者下载完成后，通过以下命令加载:

```
docker load -i oneflow-manylinux2014-cuda11.2-0.1.tar.gz
```

容器启动方式不再需要手动安装 oneflow package 和 oneflow_gpt package，而是提供 OneFlow wheel 包 和 OneFlow GPT 源码路径，由脚本启动容器后自动安装。

启动命令：

```
bash scripts/pretrain_with_container.sh
```

该命令参数在 `pretrain.sh` 已有的参数之外还额外提供容器相关参数配置。用户需要配置的参数变量如下：

- `oneflow_gpt_src_dir`: OneFlow GPT 的源码路径，根据之前的 OneFlow-Benchmark repo clone 路径和 `OneFlow-Benchmark/LanguageModeling/GPT` 配置
- `python_version`: 容器中使用的 python 版本
- `image`: 镜像，上文有提，例 `oneflow-manylinux2014-cuda11.2:0.1`
- `wheel`: OneFlow wheel 包地址，上文有列下载地址，或用户手动编译打包

`pretrain_with_container.sh` 命令通过调用 oneflow_gpt tools 中的 `launch_container.py` 来完成容器启动并运行任务，这里有个额外的参数 `--extra-mount "/data"` 是为了映射数据目录，如果你的数据集路径与笔者位置相同（`/data/gpt/gpt_sample_dataset_text_document`），可以保持该配置不变，否则需要配置自己的数据集路径的映射目录。

配置完成后，即可运行命令启动测试，输出结果与上文裸机运行测试相同。

## 测试结果

所有 case 测试的结果统计如下：

| group | case | lantency | memory | throuthput(sample/sec) | Achieved teraFLOP/s per GPU | Percentage of theoretical peak FLOP/s | Achieved aggregate petaFLOP/s
| - | - | - | - | - | - | - | -
| DP |   |   |   |   |   |   |  
|  | DP_1x1x1_2_1536x16 |  406.16 (ms) |  11066 (MiB) |  4.92 |  49.43 |  40% |  0.05
|  | DP_8x1x1_16_1536x16 |  417.95 (ms) |  12404 (MiB) |  38.28 |  48.03 |  38% |  0.38
|  | DP_16x1x1_32_1536x16 |  489.67 (ms) |  12200 (MiB) |  65.35 |  41 |  33% |  0.66
|  | DP_32x1x1_64_1536x16 |  518.64 (ms) |  12208 (MiB) |  123.4 |  38.71 |  31% |  1.24
| MP |   |   |   |   |   |   |  
|  | MP_1x1x1_8_768x12 |  525.21 (ms) |  7416 (MiB) |  15.23 |  37.98 |  30% |  0.04
|  | MP_1x8x1_16_1536x16 |  642.13 (ms) |  6264 (MiB) |  24.92 |  31.26 |  25% |  0.25
|  | MP_1x16x1_16_3072x16 |  4819.29 (ms) |  8678 (MiB) |  3.32 |  7.25 |  6% |  0.12
|  | MP_1x32x1_16_3072x32 |  9505.83 (ms) |  11256 (MiB) |  1.68 |  3.57 |  3% |  0.11
| 2D |   |   |   |   |   |   |  
|  | 2D_8x1x1_16_1536x16 |  417.90 (ms) |  12396 (MiB) |  38.29 |  48.04 |  38% |  0.38
|  | 2D_1x8x1_16_1536x16 |  644.4 (ms) |  6264 (MiB) |  24.83 |  31.15 |  25% |  0.25
|  | 2D_2x4x1_16_1536x16 |  539.69 (ms) |  6366 (MiB) |  29.65 |  37.2 |  30% |  0.3
|  | 2D_4x2x1_16_1536x16 |  543.88 (ms) |  7368 (MiB) |  29.42 |  36.91 |  30% |  0.3
|  | 2D_2x8x1_16_2304x24 |  1268.53 (ms) |  6998 (MiB) |  12.61 |  23.81 |  19% |  0.38
|  | 2D_4x8x1_32_2304x24 |  1480.34 (ms) |  6996 (MiB) |  21.62 |  20.41 |  16% |  0.65
| PP |   |   |   |   |   |   |  
|  | DP_PP_8x1x4_512_1536x16 |  4006.31 (ms) |  8712 (MiB) |  127.8 |  40.09 |  32% |  1.28
|  | MP_PP_1x8x4_512_2304x24 |  19196.65 (ms) |  15670 (MiB) |  26.67 |  25.18 |  20% |  0.81
|  | 2D_PP_2x4x4_512_2304x24 |  13117.18 (ms) |  11958 (MiB) |  39.03 |  36.85 |  29% |  1.18
|  | 2D_PP_2x8x2_512_2304x24 |  14375.35 (ms) |  10708 (MiB) |  35.62 |  33.62 |  27% |  1.08

测试过程中所有 case 的输出 log 如下：

| group | case | oneflow_logs
| - | - | -
| DP | | 
|    | DP_1x1x1_2_1536x16   | [pretrain_gpt_1n1d_dp1_mp1_pp1_mbz2_gbz2_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_1n1d_dp1_mp1_pp1_mbz2_gbz2_s2048_l16_h1536_nh16.log)
|    | DP_8x1x1_16_1536x16  | [pretrain_gpt_1n8d_dp8_mp1_pp1_mbz2_gbz16_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_1n8d_dp8_mp1_pp1_mbz2_gbz16_s2048_l16_h1536_nh16.log)
|    | DP_16x1x1_32_1536x16 | [pretrain_gpt_2n8d_dp16_mp1_pp1_mbz2_gbz32_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_2n8d_dp16_mp1_pp1_mbz2_gbz32_s2048_l16_h1536_nh16.log)
|    | DP_32x1x1_64_1536x16 | [pretrain_gpt_4n8d_dp32_mp1_pp1_mbz2_gbz64_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_4n8d_dp32_mp1_pp1_mbz2_gbz64_s2048_l16_h1536_nh16.log)
| MP | | 
|    | MP_1x1x1_8_768x12    | [pretrain_gpt_1n1d_dp1_mp1_pp1_mbz8_gbz8_s2048_l12_h768_nh12.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_1n1d_dp1_mp1_pp1_mbz8_gbz8_s2048_l12_h768_nh12.log)
|    | MP_1x8x1_16_1536x16  | [pretrain_gpt_1n8d_dp1_mp8_pp1_mbz16_gbz16_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_1n8d_dp1_mp8_pp1_mbz16_gbz16_s2048_l16_h1536_nh16.log)
|    | MP_1x16x1_16_3072x16 | [pretrain_gpt_2n8d_dp1_mp16_pp1_mbz16_gbz16_s2048_l16_h3072_nh32.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_2n8d_dp1_mp16_pp1_mbz16_gbz16_s2048_l16_h3072_nh32.log)
|    | MP_1x32x1_16_3072x32 | [pretrain_gpt_4n8d_dp1_mp32_pp1_mbz16_gbz16_s2048_l32_h3072_nh32.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_4n8d_dp1_mp32_pp1_mbz16_gbz16_s2048_l32_h3072_nh32.log)
| 2D | | 
|    | 2D_8x1x1_16_1536x16  | [pretrain_gpt_1n8d_dp8_mp1_pp1_mbz2_gbz16_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_1n8d_dp8_mp1_pp1_mbz2_gbz16_s2048_l16_h1536_nh16.log)
|    | 2D_1x8x1_16_1536x16  | [pretrain_gpt_1n8d_dp1_mp8_pp1_mbz16_gbz16_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_1n8d_dp1_mp8_pp1_mbz16_gbz16_s2048_l16_h1536_nh16.log)
|    | 2D_2x4x1_16_1536x16  | [pretrain_gpt_1n8d_dp2_mp4_pp1_mbz8_gbz16_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_1n8d_dp2_mp4_pp1_mbz8_gbz16_s2048_l16_h1536_nh16.log)
|    | 2D_4x2x1_16_1536x16  | [pretrain_gpt_1n8d_dp4_mp2_pp1_mbz4_gbz16_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_1n8d_dp4_mp2_pp1_mbz4_gbz16_s2048_l16_h1536_nh16.log)
|    | 2D_2x8x1_16_2304x24  | [pretrain_gpt_2n8d_dp2_mp8_pp1_mbz8_gbz16_s2048_l24_h2304_nh24.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_2n8d_dp2_mp8_pp1_mbz8_gbz16_s2048_l24_h2304_nh24.log)
|    | 2D_4x8x1_32_2304x24  | [pretrain_gpt_4n8d_dp4_mp8_pp1_mbz8_gbz32_s2048_l24_h2304_nh24.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_4n8d_dp4_mp8_pp1_mbz8_gbz32_s2048_l24_h2304_nh24.log)
| PP | |  
|    | DP_PP_8x1x4_512_1536x16 | [pretrain_gpt_4n8d_dp8_mp1_pp4_mbz2_gbz512_s2048_l16_h1536_nh16.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_4n8d_dp8_mp1_pp4_mbz2_gbz512_s2048_l16_h1536_nh16.log)
|    | MP_PP_1x8x4_512_2304x24 | [pretrain_gpt_4n8d_dp1_mp8_pp4_mbz16_gbz512_s2048_l24_h2304_nh24.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_4n8d_dp1_mp8_pp4_mbz16_gbz512_s2048_l24_h2304_nh24.log)
|    | 2D_PP_2x4x4_512_2304x24 | [pretrain_gpt_4n8d_dp2_mp4_pp4_mbz8_gbz512_s2048_l24_h2304_nh24.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_4n8d_dp2_mp4_pp4_mbz8_gbz512_s2048_l24_h2304_nh24.log)
|    | 2D_PP_2x8x2_512_2304x24 | [pretrain_gpt_4n8d_dp2_mp8_pp2_mbz8_gbz512_s2048_l24_h2304_nh24.log](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/perf_test_logs/pretrain_gpt_4n8d_dp2_mp8_pp2_mbz8_gbz512_s2048_l24_h2304_nh24.log)
