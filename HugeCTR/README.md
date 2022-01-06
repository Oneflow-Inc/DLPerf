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
│   ├── 300k_iters.sh # 300k iterations test, display loss and auc every 1000 iterations.
│   ├── 500_iters.sh # 500 iterations test, display loss and auc every iteration.
│   ├── bsz_x2.sh # Batch Size Double Test
│   ├── fix_bsz_per_device.sh # test with different number of devices and fixing batch size per device
│   ├── fix_total_bsz.sh # test with different number of devices and fixing total batch size
│   ├── gpu_memory_usage.py # log maximum GPU device memory usage during testing 
└── tools
    └── extract_hugectr_logs.py
    └── extract_losses_aucs.sh
```
## Benchmark Test Cases

This report summarized HugeCTR test on 1 nodes with 8 x Tesla V100 on Dec 2021

### Test Environment
- 1 nodes with Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz  ($ cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c*)
- Memory 384G ($ cat /proc/meminfo)
- Ubuntu 20.04.3 LTS  ($  cat /etc/issue/) (GNU/Linux 5.4.0-26-generic x86_64)   ($  uname -a)
- CUDA Version: 11.4  ($  nvcc -V), Driver Version: 470.57.02  ($  cat /proc/driver/nvidia/version)
- HugeCTR version: 3.2
- `nvidia-smi topo -m`

```
		GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	mlx5_0	mlx5_1	CPU Affinity    NUMA Affinity
GPU0	 X 		NV1	    NV2	    NV1	 	SYS		SYS		SYS		NV2		NODE	SYS		0-23,48-71		0
GPU1	NV1	 	X 		NV1		NV2		SYS		SYS		NV2		SYS		NODE	SYS		0-23,48-71		0
GPU2	NV2		NV1	 	X 		NV2		SYS		NV1		SYS		SYS		PIX		SYS		0-23,48-71		0
GPU3	NV1		NV2		NV2		X 		NV1		SYS		SYS		SYS		PIX		SYS		0-23,48-71		0
GPU4	SYS		SYS		SYS		NV1		X 		NV2		NV2		NV1		SYS		NODE	24-47,72-95		1
GPU5	SYS		SYS		NV1		SYS		NV2	 	X 		NV1		NV2		SYS		NODE	24-47,72-95		1
GPU6	SYS		NV2		SYS		SYS		NV2		NV1	 	X 		NV1		SYS		PIX		24-47,72-95		1
GPU7	NV2		SYS		SYS		SYS		NV1		NV2		NV1	 	X 		SYS		PIX		24-47,72-95		1
mlx5_0	NODE	NODE	PIX		PIX		SYS		SYS		SYS		SYS	 	X 		SYS		
mlx5_1	SYS		SYS		SYS		SYS		NODE	NODE	PIX		PIX		SYS	 	X 		

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

To show the tendency of the loss and AUC curves clearly, the total training batch size is set to 512 which is a small value compared to the WDL industry production scenario. Validation should be performed every training step.

### 300,000 Iterations Test
This test aims to show the training loss profile and validation area under the ROC Curve(AUC) during 300,000 steps testing. Compare to 500 iters test, we print loss and AUC every 1000 steps, it will bring us a long period view of loss and AUC curves.

### Fixed Total Batch Size Test
This test will keep the total batch size as a constant value(default is 16384), each test case adopts a different number of GPU devices, such as 1, 2, 4, 8, 16, 32.

Latency and GPU device memory usage should be recorded in this test.

### Fixed Batch Size per Device Test
This test will keep batch size per device as a constant value(default is 16384), each test case adopts a different number of GPU devices, such as 1, 2, 4, 8, 16, 32, the total batch size is scaled up with the total number of devices.

Latency and GPU device memory usage should be recorded in this test.

### Batch Size Double Test
This test uses one GPU device, batch size of the first case is 512 while the subsequent cases are doubled, and so on. This test can be performed on single device, single node and multi-nodes.

Latency and GPU device memory usage should be recorded in this test.

### Vocabulary Size Double Test
This test will use devices as much as possible, vocabulary size of the first case is 2,000,000 while the subsequent cases are doubled, and so on. This test can be performed on single device, single node and multi-nodes.

Latency and GPU device memory usage should be recorded in this test.

## Test Results

If test with multiple devices, only the log of device 0 is recorded.

### batch size X 2 tests
All batch size double tests were performed with 2 x 1024 Hidden Fully-connected Units.

#### 1 node 1 device

| gpu  | batch_size | vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ---------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g1 | 512        | 2,359,296  | 16                      | 2                | 2.421       | 1,968            |
| n1g1 | 1024       | 2,359,296  | 16                      | 2                | 2.976       | 2,014            |
| n1g1 | 2048       | 2,359,296  | 16                      | 2                | 4.157       | 2,102            |
| n1g1 | 4096       | 2,359,296  | 16                      | 2                | 6.397       | 2,272            |
| n1g1 | 8192       | 2,359,296  | 16                      | 2                | 10.911      | 2,606            |
| n1g1 | 16384      | 2,359,296  | 16                      | 2                | 19.138      | 3,276            |
| n1g1 | 32768      | 2,359,296  | 16                      | 2                | 35.525      | 4,618            |
| n1g1 | 65536      | 2,359,296  | 16                      | 2                | 69.718      | 7,290            |
| n1g1 | 131072     | 2,359,296  | 16                      | 2                | 137.635     | 12,650           |

#### 1 node 8 devices

| gpu  | batch_size | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ------------- | ---------- | ----------- | ---------------- |
| n1g8 | 512        | 16            | 2,359,296  | 2.554       | 2,390            |
| n1g8 | 1,024      | 16            | 2,359,296  | 2.851       | 2,402            |
| n1g8 | 2,048      | 16            | 2,359,296  | 3.184       | 2,428            |
| n1g8 | 4,096      | 16            | 2,359,296  | 3.858       | 2,474            |
| n1g8 | 8,192      | 16            | 2,359,296  | 5.533       | 2,576            |
| n1g8 | 16,384     | 16            | 2,359,296  | 8.773       | 2,776            |
| n1g8 | 32,768     | 16            | 2,359,296  | 14.533      | 3,168            |
| n1g8 | 65,536     | 16            | 2,359,296  | 25.999      | 3,892            |
| n1g8 | 131,072    | 16            | 2,359,296  | 53.386      | 5,312            |
| n1g8 | 262,144    | 16            | 2,359,296  | 110.342     | 8400             |
| n1g8 | 524,288    | 16            | 2,359,296  | 242.871     | 14674            |

### vocab size X 2 tests

All vocat size double tests were performed with 7 x 1024 Hidden Fully-connected Units.

#### 1 node 1 device

| gpu  | batch_size | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ------------- | ---------- | ----------- | ---------------- |
| n1g1 | 16,384     | 16            | 3,200,000  | 62.701      | 5,584            |
| n1g1 | 16,384     | 16            | 6,400,000  | 64.563      | 6,332            |
| n1g1 | 16,384     | 16            | 12,800,000 | 68.946      | 7,830            |
| n1g1 | 16,384     | 16            | 25,600,000 | 78.496      | 10,826           |
| n1g1 | 16,384     | 16            | 51,200,000 | 99.705      | 16,818           |

#### 1 node 8 devices

| gpu  | batch_size | deep_vec_size | vocab_size | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ------------- | ---------- | ----------- | ---------------- |
| n1g8 | 16,384     | 16            | 3,200,000  | 15.046      | 3,434            |
| n1g8 | 16,384     | 16            | 6,400,000  | 16.837      | 4,180            |
| n1g8 | 16,384     | 16            | 12,800,000 | 20.871      | 5,680            |
| n1g8 | 16,384     | 16            | 25,600,000 | 29.984      | 8,676            |
| n1g8 | 16,384     | 16            | 51,200,000 | 50.746      | 14,668           |

### fixed batch size per device tests

All fix batch size per device tests were performed with 7 x 1024 Hidden Fully-connected Units.

| gpu  | batch_size | vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ---------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g1 | 16384      | 2,359,296  | 32                      | 7                | 67.037      | 5,956            |
| n1g2 | 32768      | 2,359,296  | 32                      | 7                | 76.798      | 6,282            |
| n1g4 | 65536      | 2,359,296  | 32                      | 7                | 80.530      | 7,028            |
| n1g8 | 131072     | 2,359,296  | 32                      | 7                | 96.505      | 8,512            |

### fixed total batch size tests
All fixed total batch size tests were performed with 7 x 1024 Hidden Fully-connected Units.

| gpu  | batch_size | vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ---------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g1 | 16384      | 2,359,296  | 32                      | 7                | 66.676      | 5,954            |
| n1g2 | 16384      | 2,359,296  | 32                      | 7                | 40.932      | 4,416            |
| n1g4 | 16384      | 2,359,296  | 32                      | 7                | 23.521      | 3,780            |
| n1g8 | 16384      | 2,359,296  | 32                      | 7                | 16.578      | 3,668            |