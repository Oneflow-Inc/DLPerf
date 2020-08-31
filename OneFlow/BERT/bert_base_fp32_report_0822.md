# BERT base Benchmark Test Report

本报告总结了2020-8-22进行的BERT base吞吐率评测。

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

- Data Type: Float32
- XLA: 未采用
- batch size per device: 96 64 32 24
- 测试有四组分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。

全部日志可以点击[此处](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/bert_base_logs_0822.tgz)获取。

## Test Results

### batch size = 96

本结果是`All Results`小节中，表格里 `batch_size_per_device=96`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 96                    | 149.84     | 1.00    |
| 1         | 8                | 96                    | 1158.51    | 7.73    |
| 2         | 8                | 96                    | 2257.71    | 15.07   |
| 4         | 8                | 96                    | 4455.97    | 29.75   |

### batch size = 64

本结果是`All Results`小节中，表格里 `batch_size_per_device=64`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 64                    | 149.81     | 1.00    |
| 1         | 8                | 64                    | 1138.89    | 7.60    |
| 2         | 8                | 64                    | 2189.30    | 14.61   |
| 4         | 8                | 64                    | 4310.42    | 28.77   |

### batch size = 32

本结果是`All Results`小节中，表格里 `batch_size_per_device=32`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 32                    | 145.21     | 1.00    |
| 1         | 8                | 32                    | 1042.98    | 7.18    |
| 2         | 8                | 32                    | 1890.26    | 13.02   |
| 4         | 8                | 32                    | 3715.08    | 25.59   |

### batch size = 24

本结果是`All Results`小节中，表格里 `batch_size_per_device=24`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 24                    | 140.36     | 1.00    |
| 1         | 8                | 24                    | 986.11     | 7.02    |
| 2         | 8                | 24                    | 1697.82    | 12.09   |

### All Results

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
|-----------|------------------|-----------------------|------------|
| 1         | 1                | 24                    | 140.36     |
| 1         | 1                | 24                    | 140.20     |
| 1         | 1                | 24                    | 140.35     |
| 1         | 1                | 24                    | 140.30     |
| 1         | 1                | 24                    | 140.47     |
| 1         | 1                | 24                    | 140.47     |
| 1         | 1                | 24                    | 140.42     |
| 1         | 1                | 32                    | 145.35     |
| 1         | 1                | 32                    | 145.29     |
| 1         | 1                | 32                    | 145.09     |
| 1         | 1                | 32                    | 145.24     |
| 1         | 1                | 32                    | 145.05     |
| 1         | 1                | 32                    | 145.18     |
| 1         | 1                | 32                    | 145.21     |
| 1         | 1                | 64                    | 150.00     |
| 1         | 1                | 64                    | 149.81     |
| 1         | 1                | 64                    | 149.75     |
| 1         | 1                | 64                    | 149.86     |
| 1         | 1                | 64                    | 149.79     |
| 1         | 1                | 64                    | 149.72     |
| 1         | 1                | 64                    | 149.88     |
| 1         | 1                | 96                    | 149.91     |
| 1         | 1                | 96                    | 149.84     |
| 1         | 1                | 96                    | 149.77     |
| 1         | 1                | 96                    | 149.87     |
| 1         | 1                | 96                    | 149.73     |
| 1         | 1                | 96                    | 149.83     |
| 1         | 1                | 96                    | 150.00     |
| 1         | 8                | 24                    | 993.13     |
| 1         | 8                | 24                    | 980.54     |
| 1         | 8                | 24                    | 989.21     |
| 1         | 8                | 24                    | 991.19     |
| 1         | 8                | 24                    | 983.42     |
| 1         | 8                | 24                    | 986.11     |
| 1         | 8                | 24                    | 985.86     |
| 1         | 8                | 32                    | 1045.48    |
| 1         | 8                | 32                    | 1042.81    |
| 1         | 8                | 32                    | 1047.40    |
| 1         | 8                | 32                    | 1042.98    |
| 1         | 8                | 32                    | 1040.41    |
| 1         | 8                | 32                    | 1042.04    |
| 1         | 8                | 32                    | 1046.64    |
| 1         | 8                | 64                    | 1135.61    |
| 1         | 8                | 64                    | 1135.41    |
| 1         | 8                | 64                    | 1141.54    |
| 1         | 8                | 64                    | 1138.90    |
| 1         | 8                | 64                    | 1143.61    |
| 1         | 8                | 64                    | 1138.89    |
| 1         | 8                | 64                    | 1135.60    |
| 1         | 8                | 96                    | 1160.36    |
| 1         | 8                | 96                    | 1158.25    |
| 1         | 8                | 96                    | 1158.99    |
| 1         | 8                | 96                    | 1157.86    |
| 1         | 8                | 96                    | 1157.30    |
| 1         | 8                | 96                    | 1159.67    |
| 1         | 8                | 96                    | 1158.51    |
| 2         | 8                | 24                    | 1700.84    |
| 2         | 8                | 24                    | 1697.03    |
| 2         | 8                | 24                    | 1694.80    |
| 2         | 8                | 24                    | 1695.86    |
| 2         | 8                | 24                    | 1697.82    |
| 2         | 8                | 24                    | 1701.35    |
| 2         | 8                | 24                    | 1703.85    |
| 2         | 8                | 32                    | 1887.45    |
| 2         | 8                | 32                    | 1890.26    |
| 2         | 8                | 32                    | 1876.97    |
| 2         | 8                | 32                    | 1894.01    |
| 2         | 8                | 32                    | 1891.79    |
| 2         | 8                | 32                    | 1880.45    |
| 2         | 8                | 32                    | 1891.61    |
| 2         | 8                | 64                    | 2190.21    |
| 2         | 8                | 64                    | 2186.64    |
| 2         | 8                | 64                    | 2189.27    |
| 2         | 8                | 64                    | 2184.88    |
| 2         | 8                | 64                    | 2190.97    |
| 2         | 8                | 64                    | 2189.30    |
| 2         | 8                | 64                    | 2190.64    |
| 2         | 8                | 96                    | 2255.37    |
| 2         | 8                | 96                    | 2255.93    |
| 2         | 8                | 96                    | 2258.88    |
| 2         | 8                | 96                    | 2257.71    |
| 2         | 8                | 96                    | 2259.20    |
| 2         | 8                | 96                    | 2257.70    |
| 2         | 8                | 96                    | 2259.36    |
| 4         | 8                | 32                    | 3715.24    |
| 4         | 8                | 32                    | 3697.30    |
| 4         | 8                | 32                    | 3717.89    |
| 4         | 8                | 32                    | 3707.88    |
| 4         | 8                | 32                    | 3715.08    |
| 4         | 8                | 32                    | 3694.38    |
| 4         | 8                | 32                    | 3720.52    |
| 4         | 8                | 48                    | 4103.63    |
| 4         | 8                | 64                    | 4315.05    |
| 4         | 8                | 64                    | 4303.38    |
| 4         | 8                | 64                    | 4310.42    |
| 4         | 8                | 64                    | 4312.80    |
| 4         | 8                | 64                    | 4305.71    |
| 4         | 8                | 64                    | 4310.55    |
| 4         | 8                | 64                    | 4308.84    |
| 4         | 8                | 96                    | 4449.85    |
| 4         | 8                | 96                    | 4456.82    |
| 4         | 8                | 96                    | 4460.17    |
| 4         | 8                | 96                    | 4454.99    |
| 4         | 8                | 96                    | 4455.97    |
| 4         | 8                | 96                    | 4451.41    |
| 4         | 8                | 96                    | 4458.06    |
