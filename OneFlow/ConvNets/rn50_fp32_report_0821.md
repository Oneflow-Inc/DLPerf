# ResNet50-V1.5 Benchmark Test Report
本报告总结了2020-8-21进行的ResNet50-V1.5吞吐率评测的结果。

## Test Environment
所有的测试都是在4台配置8卡V100-SXM2-16GB的服务器中进行，主要应软件配置如下：
- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- 48 CPU(s), Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
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
- 测试有四组分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中间结果作为最后结果。

全部日志可以点击[此处](https://oneflow-public.oss-cn-beijing.aliyuncs.com/DLPerf/logs/OneFlow/rn50_logs_0821.tgz)获取。

## Finial Result
本结果是`All Results`中提取出来的结果，从每一组7个结果中提取的中间值。
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup | 
| -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 128 | 384.9 | 1.00  | 
| 1 | 8 | 128 | 2969.5 | 7.71  | 
| 2 | 8 | 128 | 5906.6 | 15.35  | 
| 4 | 8 | 128 | 11711.2 | 30.43  | 


## All Results
下表经过`extract_cnn_result.py`脚本抽取。共用不同节点和GPU设备数的4组结果，每组进行7次测试。
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
| -------- | -------- | -------- | -------- |
| 1 | 1 | 128 | 383.8 |
| 1 | 1 | 128 | 384.9 |
| 1 | 1 | 128 | 385.1 |
| 1 | 1 | 128 | 384.9 |
| 1 | 1 | 128 | 383.7 |
| 1 | 1 | 128 | 384.9 |
| 1 | 1 | 128 | 382.3 |
| 1 | 8 | 128 | 2961.6 |
| 1 | 8 | 128 | 2961.6 |
| 1 | 8 | 128 | 2969.5 |
| 1 | 8 | 128 | 2990.0 |
| 1 | 8 | 128 | 3006.3 |
| 1 | 8 | 128 | 2985.9 |
| 1 | 8 | 128 | 2967.2 |
| 2 | 8 | 128 | 5915.6 |
| 2 | 8 | 128 | 5921.0 |
| 2 | 8 | 128 | 5862.3 |
| 2 | 8 | 128 | 5974.3 |
| 2 | 8 | 128 | 5906.6 |
| 2 | 8 | 128 | 5826.5 |
| 2 | 8 | 128 | 5859.8 |
| 4 | 8 | 128 | 11651.9 |
| 4 | 8 | 128 | 11797.7 |
| 4 | 8 | 128 | 11613.3 |
| 4 | 8 | 128 | 11746.5 |
| 4 | 8 | 128 | 11749.2 |
| 4 | 8 | 128 | 11682.2 |
| 4 | 8 | 128 | 11711.2 |
