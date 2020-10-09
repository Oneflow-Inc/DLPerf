# BERT base Benchmark Test Report

本报告总结了OneFlow v0.2下BERT base FP32的吞吐率评测。

## Test Environment

所有的测试都是在4台配置8张V100-SXM2-16GB GPU的服务器中进行，主要硬软件配置如下：

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
- OneFlow Benchmark仓库: 当前最新master，对应commit：[638d8f42aa064486a4268a388f2bd1e9762c98b3](https://github.com/Oneflow-Inc/OneFlow-Benchmark/commit/638d8f42aa064486a4268a388f2bd1e9762c98b3)
- Data Type: Float32
- XLA: 未采用
- batch size per device: 96 64 32
- 测试有四组分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。

全部日志可以点击[bert_base_fp32_b32_64_96_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/bert_base_fp32_b32_64_96_logs.tar)

## Test Results

### batch size = 96

本结果是`All Results/OneFlow v0.2 logs`小节中，表格里 `batch_size_per_device=96`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 96 | 156.02 | 1.00 |
| 1 | 8 | 96 | 1201.70 | 7.70 |
| 2 | 8 | 96 | 2352.92 | 15.08 |
| 4 | 8 | 96 | 4664.10 | 29.89 |

### batch size = 64

本结果是`All Results/OneFlow v0.2 logs`小节中，表格里 `batch_size_per_device=64`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 64 | 156.23 | 1.00 |
| 1 | 8 | 64 | 1191.13 | 7.62 |
| 2 | 8 | 64 | 2297.08 | 14.70 |
| 4 | 8 | 64 | 4519.10 | 28.93 |


### batch size = 32

本结果是`All Results/0822 logs`小节中，表格里 `batch_size_per_device=32`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 32 | 152.89 | 1.00 |
| 1 | 8 | 32 | 1105.55 | 7.23 |
| 2 | 8 | 32 | 2015.78 | 13.18 |
| 4 | 8 | 32 | 3689.80 | 24.13 |

### All Results
#### OneFlow v0.2 logs
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
|-----------|------------------|-----------------------|------------|
| 1 | 1 | 24 | 148.21 |
| 1 | 1 | 24 | 148.14 |
| 1 | 1 | 24 | 147.95 |
| 1 | 1 | 24 | 148.01 |
| 1 | 1 | 24 | 148.02 |
| 1 | 1 | 24 | 148.01 |
| 1 | 1 | 24 | 147.87 |
| 1 | 1 | 32 | 152.93 |
| 1 | 1 | 32 | 153.04 |
| 1 | 1 | 32 | 152.88 |
| 1 | 1 | 32 | 152.87 |
| 1 | 1 | 32 | 152.89 |
| 1 | 1 | 32 | 152.92 |
| 1 | 1 | 32 | 152.86 |
| 1 | 1 | 64 | 156.23 |
| 1 | 1 | 64 | 156.27 |
| 1 | 1 | 64 | 156.15 |
| 1 | 1 | 64 | 156.29 |
| 1 | 1 | 64 | 156.27 |
| 1 | 1 | 64 | 156.15 |
| 1 | 1 | 64 | 156.12 |
| 1 | 1 | 96 | 156.48 |
| 1 | 1 | 96 | 156.11 |
| 1 | 1 | 96 | 156.02 |
| 1 | 1 | 96 | 155.95 |
| 1 | 1 | 96 | 155.86 |
| 1 | 1 | 96 | 155.91 |
| 1 | 1 | 96 | 156.03 |
| 1 | 2 | 32 | 278.84 |
| 1 | 2 | 32 | 278.34 |
| 1 | 2 | 32 | 277.30 |
| 1 | 2 | 32 | 279.29 |
| 1 | 2 | 32 | 277.34 |
| 1 | 2 | 32 | 281.26 |
| 1 | 2 | 32 | 279.23 |
| 1 | 2 | 64 | 301.16 |
| 1 | 2 | 64 | 300.52 |
| 1 | 2 | 64 | 299.16 |
| 1 | 2 | 64 | 301.78 |
| 1 | 2 | 64 | 300.71 |
| 1 | 2 | 64 | 301.04 |
| 1 | 2 | 64 | 300.98 |
| 1 | 2 | 96 | 302.53 |
| 1 | 2 | 96 | 303.04 |
| 1 | 2 | 96 | 304.36 |
| 1 | 2 | 96 | 301.49 |
| 1 | 2 | 96 | 302.42 |
| 1 | 2 | 96 | 302.79 |
| 1 | 2 | 96 | 303.79 |
| 1 | 4 | 32 | 563.92 |
| 1 | 4 | 32 | 561.76 |
| 1 | 4 | 32 | 561.97 |
| 1 | 4 | 32 | 557.58 |
| 1 | 4 | 32 | 558.30 |
| 1 | 4 | 32 | 554.22 |
| 1 | 4 | 32 | 555.89 |
| 1 | 4 | 64 | 600.40 |
| 1 | 4 | 64 | 599.99 |
| 1 | 4 | 64 | 594.65 |
| 1 | 4 | 64 | 599.88 |
| 1 | 4 | 64 | 599.62 |
| 1 | 4 | 64 | 595.02 |
| 1 | 4 | 64 | 595.23 |
| 1 | 4 | 96 | 607.87 |
| 1 | 4 | 96 | 602.46 |
| 1 | 4 | 96 | 605.07 |
| 1 | 4 | 96 | 599.37 |
| 1 | 4 | 96 | 601.39 |
| 1 | 4 | 96 | 602.10 |
| 1 | 4 | 96 | 604.90 |
| 1 | 8 | 32 | 1109.53 |
| 1 | 8 | 32 | 1110.02 |
| 1 | 8 | 32 | 1101.57 |
| 1 | 8 | 32 | 1106.28 |
| 1 | 8 | 32 | 1105.55 |
| 1 | 8 | 32 | 1105.54 |
| 1 | 8 | 32 | 1104.84 |
| 1 | 8 | 64 | 1192.44 |
| 1 | 8 | 64 | 1192.55 |
| 1 | 8 | 64 | 1187.56 |
| 1 | 8 | 64 | 1191.13 |
| 1 | 8 | 64 | 1192.98 |
| 1 | 8 | 64 | 1190.06 |
| 1 | 8 | 64 | 1190.36 |
| 1 | 8 | 96 | 1201.70 |
| 1 | 8 | 96 | 1201.41 |
| 1 | 8 | 96 | 1201.60 |
| 1 | 8 | 96 | 1199.54 |
| 1 | 8 | 96 | 1203.38 |
| 1 | 8 | 96 | 1203.42 |
| 1 | 8 | 96 | 1202.60 |
| 2 | 8 | 32 | 2015.78 |
| 2 | 8 | 32 | 2016.18 |
| 2 | 8 | 32 | 2017.96 |
| 2 | 8 | 32 | 2014.81 |
| 2 | 8 | 32 | 2014.31 |
| 2 | 8 | 32 | 2017.89 |
| 2 | 8 | 32 | 2015.46 |
| 2 | 8 | 64 | 2302.65 |
| 2 | 8 | 64 | 2288.98 |
| 2 | 8 | 64 | 2292.04 |
| 2 | 8 | 64 | 2300.60 |
| 2 | 8 | 64 | 2297.08 |
| 2 | 8 | 64 | 2295.73 |
| 2 | 8 | 64 | 2298.90 |
| 2 | 8 | 96 | 2354.22 |
| 2 | 8 | 96 | 2349.18 |
| 2 | 8 | 96 | 2352.92 |
| 2 | 8 | 96 | 2354.00 |
| 2 | 8 | 96 | 2349.83 |
| 2 | 8 | 96 | 2351.09 |
| 2 | 8 | 96 | 2357.86 |
| 4 | 8 | 32 | 3683.52 |
| 4 | 8 | 32 | 3698.98 |
| 4 | 8 | 32 | 3688.59 |
| 4 | 8 | 32 | 3681.68 |
| 4 | 8 | 32 | 3696.08 |
| 4 | 8 | 32 | 3689.80 |
| 4 | 8 | 32 | 3708.71 |
| 4 | 8 | 64 | 4516.97 |
| 4 | 8 | 64 | 4519.32 |
| 4 | 8 | 64 | 4519.10 |
| 4 | 8 | 64 | 4511.04 |
| 4 | 8 | 64 | 4516.49 |
| 4 | 8 | 64 | 4519.92 |
| 4 | 8 | 64 | 4523.15 |
| 4 | 8 | 96 | 4666.72 |
| 4 | 8 | 96 | 4662.40 |
| 4 | 8 | 96 | 4659.62 |
| 4 | 8 | 96 | 4655.22 |
| 4 | 8 | 96 | 4664.10 |
| 4 | 8 | 96 | 4666.90 |
| 4 | 8 | 96 | 4666.04 |
