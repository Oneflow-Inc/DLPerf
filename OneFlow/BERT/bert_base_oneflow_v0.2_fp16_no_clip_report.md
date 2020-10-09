# BERT base Benchmark Test Report

本报告总结了OneFlow v0.2下对BERT base FP16在clip_gradient=None的情况下进行的吞吐率评测。

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
- OneFlow Benchmark仓库: 当前最新master，对应commit：[638d8f42aa064486a4268a388f2bd1e9762c98b3](https://github.com/Oneflow-Inc/OneFlow-Benchmark/commit/638d8f42aa064486a4268a388f2bd1e9762c98b3)
- 设置clip_gradient=None，[详情](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/638d8f42aa064486a4268a388f2bd1e9762c98b3/LanguageModeling/BERT/util.py#L171)
- Data Type: Float16
- XLA: 未采用
- batch size per device: 64 
- 测试有四组分别使用单机单卡、单机8卡、2机16卡、4机32卡进行测试，每组测试7次，选取这7次数据中的中位数作为最后结果。

全部日志可以点击[bert_base_fp16_b64_no_clip_logs.tar](http://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow_test_log/oneflow_0.2/DLPerf/bert_base_fp16_b160_128_64_logs.tar)获取。

## Test Results

### batch size = 64

本结果是`All Results/OneFlow v0.2 logs`小节中，表格里 `batch_size_per_device=64`提取出来的结果，从每一组7个结果中提取中位数作为最终结果。

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
|-----------|------------------|-----------------------|------------|---------|
| 1 | 1 | 64 | 552.48 | 1.00 |
| 1 | 8 | 64 | 3897.19 | 7.05 |
| 2 | 8 | 64 | 6669.93 | 12.07 |
| 4 | 8 | 64 | 11195.72 | 20.26 |




### All Results
#### OneFlow v0.2 logs
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput |
|-----------|------------------|-----------------------|------------|
| 1 | 1 | 64 | 552.83 |
| 1 | 1 | 64 | 553.71 |
| 1 | 1 | 64 | 552.42 |
| 1 | 1 | 64 | 552.54 |
| 1 | 1 | 64 | 552.32 |
| 1 | 1 | 64 | 552.05 |
| 1 | 1 | 64 | 552.48 |
| 1 | 8 | 64 | 3926.32 |
| 1 | 8 | 64 | 3896.97 |
| 1 | 8 | 64 | 3869.53 |
| 1 | 8 | 64 | 3897.19 |
| 1 | 8 | 64 | 3913.62 |
| 1 | 8 | 64 | 3901.32 |
| 1 | 8 | 64 | 3892.35 |
| 2 | 8 | 64 | 6662.25 |
| 2 | 8 | 64 | 6676.74 |
| 2 | 8 | 64 | 6657.59 |
| 2 | 8 | 64 | 6654.27 |
| 2 | 8 | 64 | 6681.33 |
| 2 | 8 | 64 | 6669.93 |
| 2 | 8 | 64 | 6722.57 |
| 4 | 8 | 64 | 11149.09 |
| 4 | 8 | 64 | 11238.40 |
| 4 | 8 | 64 | 11201.16 |
| 4 | 8 | 64 | 11190.28 |
| 4 | 8 | 64 | 11183.92 |
| 4 | 8 | 64 | 11237.83 |
