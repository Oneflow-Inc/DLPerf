# BERT base Benchmark Test Report

本报告总结了OneFlow v0.4.0 下BERT base 混合精度开启dynamic loss scale 的评测结果。

## Test Environment

所有的测试都是在4台配置8张V100-SXM2-16GB GPU的服务器中进行，主要硬软件配置如下：

- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux  4.4.0-206-generic x86_64)
- CUDA Version: 10.2, Driver Version: 460.67
- OneFlow: [v0.4.0@325160b](https://github.com/Oneflow-Inc/oneflow/tree/325160bcfb786b166b063e669aea345fadee2da7)
- OneFlow-Benchmark: [c9a9342](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/c9a9342a40ff42c55da928a081b6d9c84a489594)
- `nvidia-smi topo -m`

  ```
          GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  CPU Affinity    NUMA Affinity
  GPU0     X      NV1     NV1     NV2     NV2     SYS     SYS     SYS     NODE    0-11,24-35      0
  GPU1    NV1      X      NV2     NV1     SYS     NV2     SYS     SYS     NODE    0-11,24-35      0
  GPU2    NV1     NV2      X      NV2     SYS     SYS     NV1     SYS     PIX     0-11,24-35      0
  GPU3    NV2     NV1     NV2      X      SYS     SYS     SYS     NV1     PIX     0-11,24-35      0
  GPU4    NV2     SYS     SYS     SYS      X      NV1     NV1     NV2     SYS     12-23,36-47     1
  GPU5    SYS     NV2     SYS     SYS     NV1      X      NV2     NV1     SYS     12-23,36-47     1
  GPU6    SYS     SYS     NV1     SYS     NV1     NV2      X      NV2     SYS     12-23,36-47     1
  GPU7    SYS     SYS     SYS     NV1     NV2     NV1     NV2      X      SYS     12-23,36-47     1
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



## Test Results

### FP16 with clip

- ### batch size = 160

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 160 | 625.63 | 1.00 |
| 1 | 8 | 160 | 4573.62 | 7.31 |
| 2 | 8 | 160 | 8548.44 | 13.66 |
| 4 | 8 | 160 | 15955.70 | 25.50 |

- ### batch size = 128

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 128 | 616.25 | 1.00 |
| 1 | 8 | 128 | 4440.89 | 7.21 |
| 2 | 8 | 128 | 8233.78 | 13.36 |
| 4 | 8 | 128 | 14160.75 | 22.98 |

- ### batch size = 64 

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 64 | 562.91 | 1.00 |
| 1 | 8 | 64 | 3750.87 | 6.66 |
| 2 | 8 | 64 | 6301.08 | 11.19 |
| 4 | 8 | 64 | 9445.96 | 16.78 |

全部日志可以点击[bert_dls_fp16_logs.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.4.0/bert_dls_fp16_logs.zip)获取。

