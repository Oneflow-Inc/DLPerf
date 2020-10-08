# OneFlow ResNet50-V1.5 Benchmark Test Report

本报告总结了OneFlow v0.2版本下的ResNet50-V1.5 评测结果。

## Test Environment

所有的测试都是在4台配置了8张 V100-SXM2-16GB GPU的服务器中进行，主要硬软件配置如下：

- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
- OneFlow: v0.1.9 
- OneFlow-Benchmark: master@8a78044
- `nvidia-smi topo -m`

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

## Test Descriptions

- OneFlow版本: v0.2，对应commit: [64c20462f245b5cbef4230a62fa06edff85411b3](https://github.com/Oneflow-Inc/oneflow/commit/64c20462f245b5cbef4230a62fa06edff85411b3)
- OneFlow Benchmark仓库: [dev_test_oneflow_0.2_perf分支](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/dev_test_oneflow_0.2_perf)
- Data Type: Float16
- XLA: 未采用
- batch size per device: 256
- 测试共有四组，分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。

全部日志可以点击[resnet50_fp16_256_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/resnet50_fp16_256_logs.tar)获取。

## Finial Results

本结果是`All Results`中提取出来的结果，从每组7次的数据中提取中位数作为最后结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 256 | 1472.72 | 1.00 |
| 1 | 8 | 256 | 10629.32 | 7.22 |
| 2 | 8 | 256 | 17920.40 | 12.17 |
| 4 | 8 | 256 | 33141.02 | 22.50 |


## All Results

表格展示了每次测试的全部结果，由`extract_cnn_result.py`脚本提取得到。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
|-----------|------------------|-----------------------|------------|
| 1 | 1 | 256 | 1473.65 |
| 1 | 1 | 256 | 1476.69 |
| 1 | 1 | 256 | 1472.72 |
| 1 | 1 | 256 | 1472.20 |
| 1 | 1 | 256 | 1475.18 |
| 1 | 1 | 256 | 1470.65 |
| 1 | 1 | 256 | 1472.15 |
| 1 | 8 | 256 | 10599.79 |
| 1 | 8 | 256 | 10746.83 |
| 1 | 8 | 256 | 10719.24 |
| 1 | 8 | 256 | 10629.32 |
| 1 | 8 | 256 | 10666.29 |
| 1 | 8 | 256 | 10618.26 |
| 1 | 8 | 256 | 10563.47 |
| 2 | 8 | 256 | 18090.06 |
| 2 | 8 | 256 | 18035.63 |
| 2 | 8 | 256 | 17677.50 |
| 2 | 8 | 256 | 17807.78 |
| 2 | 8 | 256 | 17973.23 |
| 2 | 8 | 256 | 17896.54 |
| 2 | 8 | 256 | 17920.40 |
| 4 | 8 | 256 | 31264.23 |
| 4 | 8 | 256 | 33208.78 |
| 4 | 8 | 256 | 33624.40 |
| 4 | 8 | 256 | 33141.02 |
| 4 | 8 | 256 | 30883.88 |
| 4 | 8 | 256 | 33307.94 |
| 4 | 8 | 256 | 32486.10 |
