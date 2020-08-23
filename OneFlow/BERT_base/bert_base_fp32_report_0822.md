# ResNet50-V1.5 Benchmark Test Report
本报告总结了2020-8-22进行的BERT base吞吐率评测的结果。

## Test Environment
所有的测试都是在4台配置8卡V100-SXM2-16GB的服务器中进行，主要应软件配置如下：
- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- 48 CPU(s), Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
- OneFlow: v0.1.9 
- NCCL: 2.7.3
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
- 测试有四组分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中间结果作为最后结果。

全部日志可以点击[此处](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/bert_base_logs_0822.tgz)获取。

## Test Results
### batch size = 96
本结果是`All Results` `batch_size_per_device=96`中提取出来的结果，从每一组7个结果中提取的中间值。
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup | 
| -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 96 | 149.8 | 1.00  | 
| 1 | 8 | 96 | 1158.5 | 7.73  | 
| 2 | 8 | 96 | 2257.7 | 15.07  | 
| 4 | 8 | 96 | 4456 | 29.75  | 

### batch size = 64 
本结果是`All Results` `batch_size_per_device=64`中提取出来的结果，从每一组7个结果中提取的中间值。
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup | 
| -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 64 | 149.8 | 1.00  | 
| 1 | 8 | 64 | 1138.9 | 7.60  | 
| 2 | 8 | 64 | 2189.3 | 14.61  | 
| 4 | 8 | 64 | 4310.4 | 28.77  | 

### batch size = 32
本结果是`All Results` `batch_size_per_device=32`中提取出来的结果，从每一组7个结果中提取的中间值。
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup | 
| -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 32 | 145.2 | 1.00  | 
| 1 | 8 | 32 | 1043 | 7.18  | 
| 2 | 8 | 32 | 1890.3 | 13.02  | 
| 4 | 8 | 32 | 3715.1 | 25.59  | 

### batch size = 24
本结果是`All Results` `batch_size_per_device=24`中提取出来的结果，从每一组7个结果中提取的中间值。
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup | 
| -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 24 | 140.4 | 1.00  | 
| 1 | 8 | 24 | 986.1 | 7.02  | 
| 2 | 8 | 24 | 1697.8 | 12.09  | 

### All Results
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
| -------- | -------- | -------- | -------- |
| 1 | 1 | 96 | 149.9 |
| 1 | 1 | 96 | 149.8 |
| 1 | 1 | 96 | 149.8 |
| 1 | 1 | 96 | 149.9 |
| 1 | 1 | 96 | 149.7 |
| 1 | 1 | 96 | 149.8 |
| 1 | 1 | 96 | 150.0 |
| 1 | 8 | 96 | 1160.4 |
| 1 | 8 | 96 | 1158.3 |
| 1 | 8 | 96 | 1159.0 |
| 1 | 8 | 96 | 1157.9 |
| 1 | 8 | 96 | 1157.3 |
| 1 | 8 | 96 | 1159.7 |
| 1 | 8 | 96 | 1158.5 |
| 2 | 8 | 96 | 2255.4 |
| 2 | 8 | 96 | 2255.9 |
| 2 | 8 | 96 | 2258.9 |
| 2 | 8 | 96 | 2257.7 |
| 2 | 8 | 96 | 2259.2 |
| 2 | 8 | 96 | 2257.7 |
| 2 | 8 | 96 | 2259.4 |
| 4 | 8 | 96 | 4449.9 |
| 4 | 8 | 96 | 4456.8 |
| 4 | 8 | 96 | 4460.2 |
| 4 | 8 | 96 | 4455.0 |
| 4 | 8 | 96 | 4456.0 |
| 4 | 8 | 96 | 4451.4 |
| 4 | 8 | 96 | 4458.1 |
| 1 | 1 | 64 | 150.0 |
| 1 | 1 | 64 | 149.8 |
| 1 | 1 | 64 | 149.8 |
| 1 | 1 | 64 | 149.9 |
| 1 | 1 | 64 | 149.8 |
| 1 | 1 | 64 | 149.7 |
| 1 | 1 | 64 | 149.9 |
| 1 | 8 | 64 | 1135.6 |
| 1 | 8 | 64 | 1135.4 |
| 1 | 8 | 64 | 1141.5 |
| 1 | 8 | 64 | 1138.9 |
| 1 | 8 | 64 | 1143.6 |
| 1 | 8 | 64 | 1138.9 |
| 1 | 8 | 64 | 1135.6 |
| 2 | 8 | 64 | 2190.2 |
| 2 | 8 | 64 | 2186.6 |
| 2 | 8 | 64 | 2189.3 |
| 2 | 8 | 64 | 2184.9 |
| 2 | 8 | 64 | 2191.0 |
| 2 | 8 | 64 | 2189.3 |
| 2 | 8 | 64 | 2190.6 |
| 4 | 8 | 64 | 4315.0 |
| 4 | 8 | 64 | 4303.4 |
| 4 | 8 | 64 | 4310.4 |
| 4 | 8 | 64 | 4312.8 |
| 4 | 8 | 64 | 4305.7 |
| 4 | 8 | 64 | 4310.6 |
| 4 | 8 | 64 | 4308.8 |
| 1 | 1 | 32 | 145.4 |
| 1 | 1 | 32 | 145.3 |
| 1 | 1 | 32 | 145.1 |
| 1 | 1 | 32 | 145.2 |
| 1 | 1 | 32 | 145.1 |
| 1 | 1 | 32 | 145.2 |
| 1 | 1 | 32 | 145.2 |
| 1 | 8 | 32 | 1045.5 |
| 1 | 8 | 32 | 1042.8 |
| 1 | 8 | 32 | 1047.4 |
| 1 | 8 | 32 | 1043.0 |
| 1 | 8 | 32 | 1040.4 |
| 1 | 8 | 32 | 1042.0 |
| 1 | 8 | 32 | 1046.6 |
| 2 | 8 | 32 | 1887.4 |
| 2 | 8 | 32 | 1890.3 |
| 2 | 8 | 32 | 1877.0 |
| 2 | 8 | 32 | 1894.0 |
| 2 | 8 | 32 | 1891.8 |
| 2 | 8 | 32 | 1880.5 |
| 2 | 8 | 32 | 1891.6 |
| 4 | 8 | 32 | 3715.2 |
| 4 | 8 | 32 | 3697.3 |
| 4 | 8 | 32 | 3717.9 |
| 4 | 8 | 32 | 3707.9 |
| 4 | 8 | 32 | 3715.1 |
| 4 | 8 | 32 | 3694.4 |
| 4 | 8 | 32 | 3720.5 |
| 1 | 1 | 24 | 140.4 |
| 1 | 1 | 24 | 140.2 |
| 1 | 1 | 24 | 140.3 |
| 1 | 1 | 24 | 140.3 |
| 1 | 1 | 24 | 140.5 |
| 1 | 1 | 24 | 140.5 |
| 1 | 1 | 24 | 140.4 |
| 1 | 8 | 24 | 993.1 |
| 1 | 8 | 24 | 980.5 |
| 1 | 8 | 24 | 989.2 |
| 1 | 8 | 24 | 991.2 |
| 1 | 8 | 24 | 983.4 |
| 1 | 8 | 24 | 986.1 |
| 1 | 8 | 24 | 985.9 |
| 2 | 8 | 24 | 1700.8 |
| 2 | 8 | 24 | 1697.0 |
| 2 | 8 | 24 | 1694.8 |
| 2 | 8 | 24 | 1695.9 |
| 2 | 8 | 24 | 1697.8 |
| 2 | 8 | 24 | 1701.3 |
| 2 | 8 | 24 | 1703.8 |
| 4 | 8 | 48 | 4103.6 |