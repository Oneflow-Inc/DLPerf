# OneFlow ResNet50-V1.5 Benchmark Test Report

本报告总结了OneFlow v0.2.0 下的ResNet50-V1.5 评测结果。

## Test Environment

所有的测试都是在4台配置了8张 V100-SXM2-16GB GPU的服务器中进行，主要硬软件配置如下：

- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
- OneFlow: v0.2.0
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

- OneFlow版本: [v0.2.0](https://github.com/Oneflow-Inc/oneflow/tree/v0.2.0)
- OneFlow Benchmark仓库版本: [v0.2.0](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/v0.2.0)
- XLA: 未采用
- 测试共有四组，分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。

## Finial Results

- ### FP16

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 256                   | 1472.72    | 1.00    |
| 1         | 8                | 256                   | 10629.32   | 7.22    |
| 2         | 8                | 256                   | 17920.40   | 12.17   |
| 4         | 8                | 256                   | 33141.02   | 22.50   |

全部日志可以点击[resnet50_fp16_256_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/resnet50_fp16_256_logs.tar)获取。

- ### FP32

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 144                   | 394.87     | 1.00    |
| 2         | 8                | 144                   | 6254.94    | 15.84   |
| 4         | 8                | 144                   | 12407.59   | 31.42   |
| 1         | 1                | 128                   | 397.64     | 1.00    |
| 1         | 8                | 128                   | 3130.34    | 7.87    |
| 2         | 8                | 128                   | 6260.30    | 15.74   |
| 4         | 8                | 128                   | 12411.97   | 31.21   |
| 1         | 1                | 96                    | 394.62     | 1.00    |
| 1         | 8                | 96                    | 3095.36    | 7.84    |
| 2         | 8                | 96                    | 6141.07    |  15.56  |
| 4         | 8                | 96                    | 12162.41   | 30.82   |

全部日志可以点击[resnet50_fp32_96_128_144_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/resnet50_fp32_96_128_144_logs.tar)获取。
