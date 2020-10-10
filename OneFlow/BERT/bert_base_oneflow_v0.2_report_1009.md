# BERT base Benchmark Test Report

本报告总结了OneFlow v0.2下BERT base 的评测结果。

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

- OneFlow版本: v0.2，对应commit: [64c20462f245b5cbef4230a62fa06edff85411b3](https://github.com/Oneflow-Inc/oneflow/commit/64c20462f245b5cbef4230a62fa06edff85411b3)
- OneFlow Benchmark仓库: [cnn_oneflow_v0.2_compatible分支](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/cnn_oneflow_v0.2_compatible)
- XLA: 未采用
- 测试共有四组，分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。



## Test Results

### FP16 with clip

- ### batch size = 160

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 160                   | 605.11     | 1.00    |
| 1         | 8                | 160                   | 4381.66    | 7.24    |
| 2         | 8                | 160                   | 8075.16    | 13.34   |
| 4         | 8                | 160                   | 15724.70   | 25.99   |

- ### batch size = 128

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 128                   | 589.12     | 1.00    |
| 1         | 8                | 128                   | 4179.94    | 7.10    |
| 2         | 8                | 128                   | 7673.42    | 13.03   |
| 4         | 8                | 128                   | 14729.00   | 25.00   |

- ### batch size = 64 

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 64                    | 534.72     | 1.00    |
| 1         | 8                | 64                    | 3399.43    | 6.36    |
| 2         | 8                | 64                    | 5745.56    | 10.75   |
| 4         | 8                | 64                    | 9911.78    | 18.54   |

全部日志可以点击[bert_base_fp16_b160_128_64_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/bert_base_fp16_b160_128_64_logs.tar)获取。

### FP16 without clip

- ### batch size = 160

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
| --------- | ---------------- | --------------------- | ---------- | ------- |
| 1        | 1                 | 160                   | 613.93     | 1.00    |
| 1        | 8                 | 160                   | 4683.36    | 7.63    |
| 2        | 8                 | 160                   | 8777.57    | 14.30   |
| 4        | 8                 | 160                   | 17210.63   | 28.03   |

- ### batch size = 64

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 64                    | 552.48     | 1.00    |
| 1         | 8                | 64                    | 3897.19    | 7.05    |
| 2         | 8                | 64                    | 6669.93    | 12.07   |
| 4         | 8                | 64                    | 11195.72   | 20.26   |

全部日志可以点击[bert_base_fp16_b160_no_clip_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/bert_base_fp16_b160_no_clip_logs.tar)和[bert_base_fp16_b64_no_clip_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/bert_base_fp16_b64_no_clip_logs.tar)获取。

### FP32 with clip

- ### batch size = 96

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 96                    | 156.02     | 1.00    |
| 1         | 8                | 96                    | 1201.70    | 7.70    |
| 2         | 8                | 96                    | 2352.92    | 15.08   |
| 4         | 8                | 96                    | 4664.10    | 29.89   |

- ### batch size = 64

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 64                    | 156.23     | 1.00    |
| 1         | 8                | 64                    | 1191.13    | 7.62    |
| 2         | 8                | 64                    | 2297.08    | 14.70   |
| 4         | 8                | 64                    | 4519.10    | 28.93   |


- ### batch size = 32

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1         | 1                | 32                    | 152.89    | 1.00 |
| 1         | 8                | 32                    | 1105.55    | 7.23 |
| 2         | 8                | 32                    | 2015.78    | 13.18 |
| 4         | 8                | 32                    | 3689.80    | 24.13 |

全部日志可以点击[bert_base_fp32_b32_64_96_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/bert_base_fp32_b32_64_96_logs.tar)获取。

### FP32 with clip

- ### batch size = 96

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
| --------- | ---------------- | --------------------- | ---------- | ------- |
| 1         | 1                | 96                    | 156.25     | 1.00    |
| 1         | 8                | 96                    | 1234.65    | 7.90    |
| 2         | 8                | 96                    | 2425.97    | 15.53   |
| 4         | 8                | 96                    | 4799.64    | 30.72   |

- ### batch size = 32

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
| --------- | ---------------- | --------------------- | ---------- | ------- |
| 1         | 1                | 32                    | 153.94     | 1.00    |
| 1         | 8                | 32                    | 1194.48    | 7.76    |
| 2         | 8                | 32                    | 2181.51    | 14.17   |
| 4         | 8                | 32                    | 4019.45    | 26.11   |

全部日志可以点击[bert_base_fp32_b96_32_no_clip_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/bert_base_fp32_b96_32_no_clip_logs.tar)