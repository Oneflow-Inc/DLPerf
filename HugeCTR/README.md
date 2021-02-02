# NVIDIA HugeCTR Benchmark Test 
This folder holds NVIDIA HugeCTR Benchmark Test scripts, tools and reports.

You can refer to [HugeCTR User Guide](https://github.com/NVIDIA/HugeCTR/blob/master/docs/hugectr_user_guide.md) for additional information.

## folder structure
```
DLPerf/HugeCTR $ tree
.
├── docker
│   ├── build.sh
│   ├── Dockerfile
│   └── launch.sh
├── imgs
├── README.md
├── scripts
│   ├── run_all.sh # run all tests at once.
│   ├── 300k_iters.sh # 300k iterations test, display loss and auc every 1000 iterations.
│   ├── 500_iters.sh # 500 iterations test, display loss and auc every iteration.
│   ├── bsz_x2_1_device_run.sh # Batch Size Double Test on 1 device.
│   ├── bsz_x2_1_node_run.sh # Batch Size Double Test on 1 node.
│   ├── bsz_x2_mul_nodes_run.sh # Batch Size Double Test on multi-nodes
│   ├── fix_bsz_per_device_mul_nodes_run.sh # test with different number of devices and fixing batch size per device
│   ├── fix_total_bsz_mul_nodes_run.sh # test with different number of devices and fixing total batch size
│   ├── vocab_x2_1_device_run.sh # Vocabulary Size Double Test on 1 device.
│   ├── vocab_x2_1_node_run.s # Vocabulary Size Double Test on 1 node.
│   ├── vocab_x2_mul_nodes_run.sh # Vocabulary Size Double Test on multi-nodes.
│   ├── gpu_memory_usage.py # log maximum GPU device memory usage during testing
│   ├── gen_hugectr_conf_json.py # generate HugeCTR conf json file for testing
│   ├── wdl_2x1024.json # template file for HugeCTR conf json generator, 2x1024 units FCs 
│   └── wdl_7x1024.json # template file for HugeCTR conf json generator, 7x1024 units FCs 
└── tools
    └── extract_hugectr_logs.py
```
## Benchmark Test Cases

This report summarized HugeCTR test on 4 nodes with 8 x Tesla V100 on Oct 2020.

### Test Environment
- 4 nodes with Tesla V100-SXM2-16GB x 8 each
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
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

### 500 Iterations Test
This test aims to show the training loss convergency profile and validation area under the ROC Curve(AUC) during 500 steps testing. 

To show the tendency of the loss and AUC curves clearly, the total training batch size is set to 512, 512 is a small value compared to the WDL industry production scenario. Validation should be performed every training step.

### 300,000 Iterations Test
This test aims to show the training loss profile and validation area under the ROC Curve(AUC) during 300,000 steps testing. Compare to 500 iters test, we print loss and AUC every 1000 steps, it will bring us a long period view of loss and AUC curves.

### Fixed Total Batch Size Test
This test will keep the total batch size as a constant value(default is 16384), each test case adopts a different number of GPU devices, such as 1, 2, 4, 8, 16, 32.

Latency and GPU device memory usage should be recorded in this test.

### Fixed Batch Size per Device Test
This test will keep batch size per device as a constant value(default is 16384), each test case adopts a different number of GPU devices, such as 1, 2, 4, 8, 16, 32, the total batch size is scaled up with the total number of devices of the test case.

Latency and GPU device memory usage should be recorded in this test.

### Batch Size Double Test
This test uses one GPU device, the first case's batch size is 512, the batch size of the subsequent case is doubled, and so on. This test can be performed on single device, single node and multi-nodes.

Latency and GPU device memory usage should be recorded in this test.

### Vocabulary Size Double Test
This test uses devices as much as possible, the first case's vocabulary size is 3,200,000, the vocab size of the subsequent case is doubled, and so on. This test can be performed on single device, single node and multi-nodes.

Latency and GPU device memory usage should be recorded in this test.

## Test Results

If it is a a test with multiple devices, the log of device 0 is recorded.

### batch size X 2 tests
All batch size double tests were performed with 2 x 1024 Hidden Fully-connected Units.

#### 1 node 1 device
| gpu | batchsize  | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| -------- | -------- | -------- | -------- | -------- | -------- |
| n1g1 | 512 | 16 | 2,322,444 | 2.74 | 1,879 |
| n1g1 | 1,024 | 16 | 2,322,444 | 3.318 | 1,883 |
| n1g1 | 2,048 | 16 | 2,322,444 | 4.581 | 1,937 |
| n1g1 | 4,096 | 16 | 2,322,444 | 6.632 | 2,055 |
| n1g1 | 8,192 | 16 | 2,322,444 | 10.69 | 2,283 |
| n1g1 | 16,384 | 16 | 2,322,444 | 18.978 | 2,735 |
| n1g1 | 32,768 | 16 | 2,322,444 | 34.724 | 3,645 |
| n1g1 | 65,536 | 16 | 2,322,444 | 67.841 | 5,467 |
| n1g1 | 131,072 | 16 | 2,322,444 | 135.111 | 9,101 |

![latency](./imgs/hugectr_1node_1device_batchsize_double.png)

#### 1 node 8 devices
| gpu | batchsize  | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| -------- | -------- | -------- | -------- | -------- | -------- |
| n1g8 | 512 | 16 | 2,322,444 | 3.335 | 2,229 |
| n1g8 | 1,024 | 16 | 2,322,444 | 4.828 | 2,237 |
| n1g8 | 2,048 | 16 | 2,322,444 | 5.038 | 2,253 |
| n1g8 | 4,096 | 16 | 2,322,444 | 5.506 | 2,291 |
| n1g8 | 8,192 | 16 | 2,322,444 | 6.605 | 2,353 |
| n1g8 | 16,384 | 16 | 2,322,444 | 9.52 | 2,475 |
| n1g8 | 32,768 | 16 | 2,322,444 | 12.768 | 2,727 |
| n1g8 | 65,536 | 16 | 2,322,444 | 28.938 | 3,225 |
| n1g8 | 131,072 | 16 | 2,322,444 | 38.076 | 4,225 |
| n1g8 | 262,144 | 16 | 2,322,444 | 68.737 | 6,221 |
| n1g8 | 524,288 | 16 | 2,322,444 | 118.361 | 10,219 |

![latency](./imgs/hugectr_1node_8device_batchsize_double.png)

#### 4 nodes 32 devices
| gpu | batchsize  | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| -------- | -------- | -------- | -------- | -------- | -------- |
| n4g8 | 16,384 | 32 | 2,322,444 | 14.536 | 2,633 |
| n4g8 | 32,768 | 32 | 2,322,444 | 21.28 | 2,901 |
| n4g8 | 65,536 | 32 | 2,322,444 | 38.613 | 3,449 |
| n4g8 | 131,072 | 32 | 2,322,444 | 69.515 | 4,537 |
| n4g8 | 262,144 | 32 | 2,322,444 | 138.886 | 6,711 |
| n4g8 | 524,288 | 32 | 2,322,444 | 270.68 | 11,055 |

![latency](./imgs/hugectr_4node_32device_batchsize_double.png)

### vocab size X 2 tests
All vocat size double tests were performed with 7 x 1024 Hidden Fully-connected Units.

#### 1 node 1 device

| gpu | batchsize  | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| -------- | -------- | -------- | -------- | -------- | -------- |
| n1g1 | 16,384 | 16 | 3,200,000 | 65.664 | 4,427 |
| n1g1 | 16,384 | 16 | 6,400,000 | 67.913 | 5,177 |
| n1g1 | 16,384 | 16 | 12,800,000 | 72.729 | 6,727 |
| n1g1 | 16,384 | 16 | 25,600,000 | 82.853 | 9,825 |
| n1g1 | 16,384 | 16 | 51,200,000 | 104.458 | 16,027 |

![latency](./imgs/hugectr_1node_1device_vocabsize_double.png)

#### 1 node 8 devices

| gpu | batchsize  | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| -------- | -------- | -------- | -------- | -------- | -------- |
| n1g8 | 16,384 | 16 | 3,200,000 | 16.671 | 3,021 |
| n1g8 | 16,384 | 16 | 6,400,000 | 19.036 | 3,797 |
| n1g8 | 16,384 | 16 | 12,800,000 | 23.707 | 5,347 |
| n1g8 | 16,384 | 16 | 25,600,000 | 34.618 | 8,447 |
| n1g8 | 16,384 | 16 | 51,200,000 | 57.106 | 14,649 |

![latency](./imgs/hugectr_1node_8device_vocabsize_double.png)

#### 4 node 32 devices
| gpu | batchsize  | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| -------- | -------- | -------- | -------- | -------- | -------- |
| n4g8 | 16,384 | 32 | 3,200,000 | 21.843 | 3,217 |
| n4g8 | 16,384 | 32 | 6,400,000 | 26.375 | 4,579 |
| n4g8 | 16,384 | 32 | 12,800,000 | 36.214 | 7,299 |
| n4g8 | 16,384 | 32 | 25,600,000 | 57.718 | 12,745 |

![latency](./imgs/hugectr_4node_32device_vocabsize_double.png)

### fixed batch size per device tests
All fix batch size per device tests were performed with 7 x 1024 Hidden Fully-connected Units.

| gpu | batchsize  | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| -------- | -------- | -------- | -------- | -------- | -------- |
| n1g1 | 16,384 | 32 | 2,322,444 | 70.181 | 4,845 |
| n1g2 | 32,768 | 32 | 2,322,444 | 75.446 | 5,139 |
| n1g4 | 65,536 | 32 | 2,322,444 | 75.492 | 5,765 |
| n1g8 | 131,072 | 32 | 2,322,444 | 87.97 | 6,947 |
| n2g8 | 262,144 | 32 | 2,322,444 | 155.461 | 8,527 |
| n4g8 | 524,288 | 32 | 2,322,444 | 270.68 | 12,417 |

![latency](./imgs/hugectr_fixed_batchsize_per_device.png)

### fixed total batch size tests
All fixed total batch size tests were performed with 7 x 1024 Hidden Fully-connected Units.

| gpu | batchsize  | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| -------- | -------- | -------- | -------- | -------- | -------- |
| n1g1 | 16,384 | 32 | 2,322,444 | 70.216 | 4,845 |
| n1g2 | 16,384 | 32 | 2,322,444 | 40.928 | 3,769 |
| n1g4 | 16,384 | 32 | 2,322,444 | 23.903 | 3,351 |
| n1g8 | 16,384 | 32 | 2,322,444 | 18.817 | 3,261 |
| n2g8 | 16,384 | 32 | 2,322,444 | 22.34 | 2,817 |
| n4g8 | 16,384 | 32 | 2,322,444 | 21.093 | 2,751 |

![latency](./imgs/hugectr_fixed_total_batchsize.png)

