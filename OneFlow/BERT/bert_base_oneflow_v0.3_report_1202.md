# BERT base Benchmark Test Report

本报告总结了OneFlow v0.3.0 下BERT base 混合精度开启dynamic loss scale 的评测结果。

## Test Environment

所有的测试都是在4台配置8张V100-SXM2-16GB GPU的服务器中进行，主要硬软件配置如下：

- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
- OneFlow: v0.3.0@f4bf35f7a
- OneFlow-Benchmark: v0.3.0@854ddd06b
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

- OneFlow版本: [v0.3.0@f4bf35f7a](https://github.com/Oneflow-Inc/oneflow/tree/v0.3.0)
- OneFlow Benchmark仓库版本: [v0.3.0@854ddd06b](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/v0.3.0)
- Dynamic Loss Scale: 开启
- XLA: 未采用
- 测试共有四组，分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。



## Test Results

### FP16 with clip

- ### batch size = 160

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 160                   | 613.46     | 1.00    |
| 1         | 8                | 160                   | 4514.64    | 7.36    |
| 2         | 8                | 160                   | 8325.87    | 13.57   |
| 4         | 8                | 160                   | 16001.63   | 26.08   |

- ### batch size = 128

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 128 | 595.94 | 1.00 |
| 1 | 8 | 128 | 4313.91 | 7.24 |
| 2 | 8 | 128 | 7878.62 | 13.22 |
| 4 | 8 | 128 | 15113.94 | 25.36 |

- ### batch size = 64 

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 64 | 534.17 | 1.00 |
| 1 | 8 | 64 | 3519.61 | 6.59 |
| 2 | 8 | 64 | 5991.10 | 11.22 |
| 4 | 8 | 64 | 10026.29 | 18.77 |

全部日志可以点击[bert_dls_fp16_logs.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.3/bert_dls_fp16_logs.zip)获取。

