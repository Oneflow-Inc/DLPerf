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
- Data Type: Float32
- XLA: 未采用
- batch size per device: 96/128/144
- 测试共有四组，分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。

全部日志可以点击[resnet50_fp32_96_128_144_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/resnet50_fp32_96_128_144_logs.tar)获取。

## Finial Results

本结果是`All Results`中提取出来的结果，从每组7次的数据中提取中位数作为最后结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 144 | 394.87 | 1.00 |
| 2 | 8 | 144 | 6254.94 | 15.84 |
| 4 | 8 | 144 | 12407.59 | 31.42 |
| 1 | 1 | 128 | 397.64 | 1.00 |
| 1 | 8 | 128 | 3130.34 | 7.87 |
| 2 | 8 | 128 | 6260.30 | 15.74 |
| 4 | 8 | 128 | 12411.97 | 31.21 |
| 1 | 1 | 96 | 394.62 | 1.00 |
| 1 | 8 | 96 | 3095.36 | 7.84 |
| 2 | 8 | 96 | 6141.07 | 15.56 |
| 4 | 8 | 96 | 12162.41 | 30.82 |


## All Results

表格展示了每次测试的全部结果，由`extract_cnn_result.py`脚本提取得到。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
|-----------|------------------|-----------------------|------------|
| 1 | 1 | 128 | 398.00 |
| 1 | 1 | 128 | 396.75 |
| 1 | 1 | 128 | 398.09 |
| 1 | 1 | 128 | 397.21 |
| 1 | 1 | 128 | 397.64 |
| 1 | 1 | 128 | 397.52 |
| 1 | 1 | 128 | 399.98 |
| 1 | 1 | 144 | 395.74 |
| 1 | 1 | 144 | 394.34 |
| 1 | 1 | 144 | 394.30 |
| 1 | 1 | 144 | 396.18 |
| 1 | 1 | 144 | 395.13 |
| 1 | 1 | 144 | 391.78 |
| 1 | 1 | 144 | 394.87 |
| 1 | 1 | 96 | 396.08 |
| 1 | 1 | 96 | 390.98 |
| 1 | 1 | 96 | 394.63 |
| 1 | 1 | 96 | 394.62 |
| 1 | 1 | 96 | 394.82 |
| 1 | 1 | 96 | 392.67 |
| 1 | 1 | 96 | 394.32 |
| 1 | 8 | 128 | 3134.08 |
| 1 | 8 | 128 | 3148.21 |
| 1 | 8 | 128 | 3119.86 |
| 1 | 8 | 128 | 3126.80 |
| 1 | 8 | 128 | 3130.34 |
| 1 | 8 | 128 | 3136.95 |
| 1 | 8 | 128 | 3120.94 |
| 1 | 8 | 96 | 3067.59 |
| 1 | 8 | 96 | 3101.52 |
| 1 | 8 | 96 | 3101.09 |
| 1 | 8 | 96 | 3087.12 |
| 1 | 8 | 96 | 3101.12 |
| 1 | 8 | 96 | 3095.36 |
| 1 | 8 | 96 | 3092.19 |
| 2 | 8 | 128 | 6262.19 |
| 2 | 8 | 128 | 6260.30 |
| 2 | 8 | 128 | 6264.03 |
| 2 | 8 | 128 | 6255.27 |
| 2 | 8 | 128 | 6254.53 |
| 2 | 8 | 128 | 6262.73 |
| 2 | 8 | 128 | 6253.00 |
| 2 | 8 | 144 | 6254.81 |
| 2 | 8 | 144 | 6254.92 |
| 2 | 8 | 144 | 6259.26 |
| 2 | 8 | 144 | 6254.96 |
| 2 | 8 | 144 | 6265.04 |
| 2 | 8 | 144 | 6250.34 |
| 2 | 8 | 96 | 6149.05 |
| 2 | 8 | 96 | 6145.07 |
| 2 | 8 | 96 | 6146.86 |
| 2 | 8 | 96 | 6135.97 |
| 2 | 8 | 96 | 6135.39 |
| 2 | 8 | 96 | 6141.07 |
| 2 | 8 | 96 | 6134.25 |
| 4 | 8 | 128 | 12394.24 |
| 4 | 8 | 128 | 12444.92 |
| 4 | 8 | 128 | 12402.12 |
| 4 | 8 | 128 | 12405.41 |
| 4 | 8 | 128 | 12413.94 |
| 4 | 8 | 128 | 12424.62 |
| 4 | 8 | 128 | 12411.97 |
| 4 | 8 | 144 | 12411.30 |
| 4 | 8 | 144 | 12396.80 |
| 4 | 8 | 144 | 12395.73 |
| 4 | 8 | 144 | 12419.39 |
| 4 | 8 | 144 | 12408.18 |
| 4 | 8 | 144 | 12407.00 |
| 4 | 8 | 96 | 12153.40 |
| 4 | 8 | 96 | 12162.41 |
| 4 | 8 | 96 | 12163.59 |
| 4 | 8 | 96 | 12182.33 |
| 4 | 8 | 96 | 12155.98 |
| 4 | 8 | 96 | 12153.02 |
| 4 | 8 | 96 | 12173.02 |
