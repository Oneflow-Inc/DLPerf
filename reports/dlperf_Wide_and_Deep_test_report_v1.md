# DLPerf Wide & Deep Test Report V1.0


## Test Environment

All tests were performed on 4 Nodes with 8x Tesla V100-SXM2-16GB GPUs, the following is the main hardware and software configuration for each:  

- 4 nodes with Tesla V100-SXM2-16GB x 8 each
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
- OneFlow: v0.2.0-83-gb16a8d42f 
- OneFlow-Benchmark: update_wdl@42c5515
- HugeCTR version: 2.2
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


## Framework & Models

| Framework | Version | Docker From |Model Sources|
| --------- | ------- | ----------- | ----------- |
|[OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.2.0)|0.2.0|             |[OneFolow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/v0.2.0/ClickThroughRate/WideDeepLearning)|
|[HugeCTR](https://github.com/NVIDIA/HugeCTR)| 2.2 ||[samples/wdl](https://github.com/NVIDIA/HugeCTR/tree/v2.2/samples/wdl)|

## Test Results
The following is a summary of the **Vocab Size X 2 Tests**, please refer to [OneFlow/ClickThroughRate/WideDeepLearning/reports](../OneFlow/ClickThroughRate/WideDeepLearning/reports) and [HugeCTR/reports](../HugeCTR/reports) for more details.

### 1 node 1 gpu, batch_size = 16384, deep_embedding_vec_size = 16, hidden_units_num = 7

| deep_vocab_size | OneFlow Latency per Iteration / ms | HugeCTR Latency per Iteration / ms | OneFlow Mem Usage / MB | HugeCTR Mem Usage / MB |
| --------------- | ---------------------------------- | ---------------------------------- | ---------------------- | ---------------------- |
| 3200000         | 56.601                             | 65.664                             | 2,557                  | 4427                   |
| 6400000         | 56.862                             | 67.913                             | 3,179                  | 5177                   |
| 12800000        | 56.964                             | 72.729                             | 4,421                  | 6727                   |
| 25600000        | 56.841                             | 82.853                             | 6,913                  | 9825                   |
| 51200000        | 56.805                             | 104.458                            | 11,891                 | 16027                  |

![img](C:\mygithub\DLPerf\reports\imgs\wdl_vecx2_1n1g_latency.png)

![](C:\mygithub\DLPerf\reports\imgs\wdl_vecx2_1n1g_mem.png)

### 1 node 8 gpu, batch_size = 16384, deep_embedding_vec_size = 16, hidden_units_num = 7

| deep_vocab_size | OneFlow Latency per Iteration / ms | HugeCTR Latency per Iteration / ms | OneFlow Mem Usage / MB | HugeCTR Mem Usage / MB |
| --------------- | ---------------------------------- | ---------------------------------- | ---------------------- | ---------------------- |
| 3200000   | 13.837 | 16.671 | 1,533  | 3,021  |
| 6400000   | 13.948 | 19.036 | 1,613  | 3,797  |
| 12800000  | 13.847 | 23.707 | 1,775  | 5,347  |
| 25600000  | 13.772 | 34.618 | 2,087  | 8,447  |
| 51200000  | 13.974 | 57.106 | 2,713  | 14,649 |
| 102400000 | 13.846 | out of memory | 3,945  | out of memory |
| 204800000 | 13.785 | out of memory | 6,435  | out of memory |
| 409600000 | 13.845 | out of memory | 11,423 | out of memory |

### 4 node 8 gpu, batch_size = 16384, deep_embedding_vec_size = 32, hidden_units_num = 7

| deep_vocab_size | OneFlow Latency per Iteration / ms | HugeCTR Latency per Iteration / ms | OneFlow Mem Usage / MB | HugeCTR Mem Usage / MB |
| --------------- | ---------------------------------- | ---------------------------------- | ---------------------- | ---------------------- |
| 3200000   | 22.414 | 21.843        | 1,115  | 3217          |
| 6400000   | 22.314 | 26.375        | 1,153  | 4579          |
| 12800000  | 22.352 | 36.214        | 1,227  | 7299          |
| 25600000  | 22.399 | 57.718        | 1,379  | 12745         |
| 51200000  | 22.31  | out of memory | 1,685  | out of memory |
| 102400000 | 22.444 | out of memory | 2,293  | out of memory |
| 204800000 | 22.403 | out of memory | 3,499  | out of memory |
| 409600000 | 22.433 | out of memory | 5,915  | out of memory |
| 819200000 | 22.407 | out of memory | 10,745 | out of memory |