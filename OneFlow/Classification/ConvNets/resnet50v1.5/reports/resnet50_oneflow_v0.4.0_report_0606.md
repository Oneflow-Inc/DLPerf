# OneFlow ResNet50-V1.5 Benchmark Test Report

本报告总结了OneFlow v0.4.0 的ResNet50-V1.5 混合精度情况下dynamic loss scale的评测结果。

## Test Environment

所有的测试都是在4台配置了8张 V100-SXM2-16GB GPU的服务器中进行，主要硬软件配置如下：

- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-206-generic x86_64)
- CUDA Version: 10.2, Driver Version: 460.67
- OneFlow: [v0.4.0@325160b](https://github.com/Oneflow-Inc/oneflow/tree/325160bcfb786b166b063e669aea345fadee2da7)
- OneFlow-Benchmark: [c9a9342](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/c9a9342a40ff42c55da928a081b6d9c84a489594)
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

- OneFlow版本: [v0.4.0@325160b](https://github.com/Oneflow-Inc/oneflow/tree/325160bcfb786b166b063e669aea345fadee2da7)
- OneFlow Benchmark仓库版本: [c9a9342](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/c9a9342a40ff42c55da928a081b6d9c84a489594)
- Dynamic Loss Scale: 开启
- XLA: 未采用
- 测试共有四组，分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。
- 设置cnns/ofrecord_util.py [num_workers=3](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/ofrecord_util.py#L88)

## Finial Results

- ### FP16

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 256 | 1452.84 | 1.00 |
| 1 | 8 | 256 | 9876.88 | 6.80 |
| 2 | 8 | 256 | 17256.82 | 11.88 |
| 4 | 8 | 256 | 28406.60  | 19.55 |

- 单机八卡情况下(1n8g)，设置设置cnns/ofrecord_util.py [num_workers=5](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/ofrecord_util.py#L88)，可以得10859的吞吐率。


全部日志可以点击[rn50_dls_fp16_256_logs.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.4.0/rn50_dls_fp16_256_logs.zip)获取。
