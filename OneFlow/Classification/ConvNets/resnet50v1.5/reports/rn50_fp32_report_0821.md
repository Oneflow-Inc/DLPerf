# OneFlow ResNet50-V1.5 Benchmark Test Report

本报告总结了2020-8-21进行的ResNet50-V1.5 评测结果。

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

- Data Type: Float32
- XLA: 未采用
- batch size per device: 128
- 测试共有四组，分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。

全部日志可以点击[此处](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/rn50_logs_0821.tgz)获取。

## Finial Results

本结果是`All Results`中提取出来的结果，从每组7次的数据中提取中位数作为最后结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 128                   | 384.85     | 1.00    |
| 1         | 8                | 128                   | 2969.45    | 7.72    |
| 2         | 8                | 128                   | 5906.55    | 15.35   |
| 4         | 8                | 128                   | 11711.18   | 30.43   |


## All Results

表格展示了每次测试的全部结果，由`extract_cnn_result.py`脚本提取得到。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
|-----------|------------------|-----------------------|------------|
| 1         | 1                | 128                   | 383.81     |
| 1         | 1                | 128                   | 384.86     |
| 1         | 1                | 128                   | 385.07     |
| 1         | 1                | 128                   | 384.91     |
| 1         | 1                | 128                   | 383.75     |
| 1         | 1                | 128                   | 384.85     |
| 1         | 1                | 128                   | 382.31     |
| 1         | 8                | 128                   | 2961.63    |
| 1         | 8                | 128                   | 2961.61    |
| 1         | 8                | 128                   | 2969.45    |
| 1         | 8                | 128                   | 2989.96    |
| 1         | 8                | 128                   | 3006.33    |
| 1         | 8                | 128                   | 2985.85    |
| 1         | 8                | 128                   | 2967.25    |
| 2         | 8                | 128                   | 5915.65    |
| 2         | 8                | 128                   | 5921.01    |
| 2         | 8                | 128                   | 5862.30    |
| 2         | 8                | 128                   | 5974.30    |
| 2         | 8                | 128                   | 5906.55    |
| 2         | 8                | 128                   | 5826.49    |
| 2         | 8                | 128                   | 5859.78    |
| 4         | 8                | 128                   | 11651.92   |
| 4         | 8                | 128                   | 11797.74   |
| 4         | 8                | 128                   | 11613.26   |
| 4         | 8                | 128                   | 11746.48   |
| 4         | 8                | 128                   | 11749.22   |
| 4         | 8                | 128                   | 11682.19   |
| 4         | 8                | 128                   | 11711.18   |
