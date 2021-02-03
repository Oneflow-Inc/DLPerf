# BERT base Benchmark Test Report

本报告总结了2020-8-22进行的BERT base吞吐率评测。另外在2020-8-29日进行了补充测试。

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
- batch size per device: 96 64 48 32 24
- 测试有四组分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。

全部日志可以点击[0822 logs](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/bert_base_logs_0822.tgz)和[0829 logs](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/of_bert_base_bsz_24_48_logs_0829.tgz)获取。

## Test Results

### batch size = 96

本结果是`All Results/0822 logs`小节中，表格里 `batch_size_per_device=96`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 96                    | 149.84     | 1.00    |
| 1         | 8                | 96                    | 1158.51    | 7.73    |
| 2         | 8                | 96                    | 2257.71    | 15.07   |
| 4         | 8                | 96                    | 4455.97    | 29.75   |

### batch size = 64

本结果是`All Results/0822 logs`小节中，表格里 `batch_size_per_device=64`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 64                    | 149.81     | 1.00    |
| 1         | 8                | 64                    | 1138.89    | 7.60    |
| 2         | 8                | 64                    | 2189.30    | 14.61   |
| 4         | 8                | 64                    | 4310.42    | 28.77   |

### batch size = 48 

本结果是`All Results/0829 logs`小节中，表格里 `batch_size_per_device=48`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 48                    | 148.01     | 1.00    |
| 1         | 8                | 48                    | 1103.01    | 7.45    |
| 2         | 8                | 48                    | 2078.66    | 14.04   |
| 4         | 8                | 48                    | 4090.99    | 27.64   |

### batch size = 32

本结果是`All Results/0822 logs`小节中，表格里 `batch_size_per_device=32`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 32                    | 145.21     | 1.00    |
| 1         | 8                | 32                    | 1042.98    | 7.18    |
| 2         | 8                | 32                    | 1890.26    | 13.02   |
| 4         | 8                | 32                    | 3715.08    | 25.59   |

### batch size = 24

本结果是`All Results/0829 logs`小节中，表格里 `batch_size_per_device=24`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 24                    | 139.98     | 1.00    |
| 1         | 8                | 24                    | 985.10     | 7.04    |
| 2         | 8                | 24                    | 1700.05    | 12.14   |
| 4         | 8                | 24                    | 3312.17    | 23.66   |

### Throughput on different batch sizes

下图展示了不同batch size情况下，1卡、8卡、16卡、32卡GPU的吞吐率，batch size越大吞吐率越高。
![Throughput on different batch sizes](imgs/of_bert_base_throughput.png)

### Speedup on different batch sizes
下图展示了不同batch size情况下，加速比曲线，batch size越大加速比越高。
![Speed on different batch sizes](imgs/of_bert_base_speedup.png)

### Latency vs Throughput on different batch sizes
![Latency vs Throughput on different batch sizes](imgs/of_bert_base_latency_throughput.png)

### All Results
#### 0822 logs
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

#### 0829 logs
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
|-----------|------------------|-----------------------|------------|
| 1         | 1                | 24                    | 140.10     |
| 1         | 1                | 24                    | 139.98     |
| 1         | 1                | 24                    | 139.97     |
| 1         | 1                | 24                    | 139.85     |
| 1         | 1                | 24                    | 140.13     |
| 1         | 1                | 24                    | 140.08     |
| 1         | 1                | 24                    | 139.98     |
| 1         | 1                | 48                    | 148.15     |
| 1         | 1                | 48                    | 147.97     |
| 1         | 1                | 48                    | 148.00     |
| 1         | 1                | 48                    | 148.01     |
| 1         | 1                | 48                    | 147.91     |
| 1         | 1                | 48                    | 148.06     |
| 1         | 1                | 48                    | 148.10     |
| 1         | 8                | 24                    | 987.23     |
| 1         | 8                | 24                    | 984.66     |
| 1         | 8                | 24                    | 982.54     |
| 1         | 8                | 24                    | 989.64     |
| 1         | 8                | 24                    | 985.10     |
| 1         | 8                | 24                    | 981.43     |
| 1         | 8                | 24                    | 988.89     |
| 1         | 8                | 48                    | 1105.21    |
| 1         | 8                | 48                    | 1097.55    |
| 1         | 8                | 48                    | 1107.74    |
| 1         | 8                | 48                    | 1105.85    |
| 1         | 8                | 48                    | 1102.99    |
| 1         | 8                | 48                    | 1101.75    |
| 1         | 8                | 48                    | 1103.01    |
| 2         | 8                | 24                    | 1693.83    |
| 2         | 8                | 24                    | 1700.05    |
| 2         | 8                | 24                    | 1701.61    |
| 2         | 8                | 24                    | 1704.38    |
| 2         | 8                | 24                    | 1701.73    |
| 2         | 8                | 24                    | 1692.67    |
| 2         | 8                | 24                    | 1699.91    |
| 2         | 8                | 48                    | 2088.27    |
| 2         | 8                | 48                    | 2073.86    |
| 2         | 8                | 48                    | 2079.72    |
| 2         | 8                | 48                    | 2077.03    |
| 2         | 8                | 48                    | 2078.66    |
| 2         | 8                | 48                    | 2077.47    |
| 2         | 8                | 48                    | 2081.43    |
| 4         | 8                | 24                    | 3294.66    |
| 4         | 8                | 24                    | 3284.88    |
| 4         | 8                | 24                    | 3317.45    |
| 4         | 8                | 24                    | 3326.39    |
| 4         | 8                | 24                    | 3312.17    |
| 4         | 8                | 24                    | 3302.04    |
| 4         | 8                | 24                    | 3318.51    |
| 4         | 8                | 48                    | 4090.99    |
| 4         | 8                | 48                    | 4079.47    |
| 4         | 8                | 48                    | 4077.71    |
| 4         | 8                | 48                    | 4086.21    |
| 4         | 8                | 48                    | 4101.22    |
| 4         | 8                | 48                    | 4102.27    |
| 4         | 8                | 48                    | 4106.70    |
