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
│   ├── core
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
| n1g1 | 512        | 2,000,000  | 16                      | 2                | 2.173       | 1,8              |
| n1g1 | 1024       | 2,000,000  | 16                      | 2                | 2.723       | 1,930            |
| n1g1 | 2048       | 2,000,000  | 16                      | 2                | 3.903       | 2,016            |
| n1g1 | 4096       | 2,000,000  | 16                      | 2                | 6.093       | 2,184            |
| n1g1 | 8192       | 2,000,000  | 16                      | 2                | 10.306      | 2,520            |
| n1g1 | 16384      | 2,000,000  | 16                      | 2                | 18.536      | 3,190            |
| n1g1 | 32768      | 2,000,000  | 16                      | 2                | 35.087      | 4,532            |
| n1g1 | 65536      | 2,000,000  | 16                      | 2                | 70.169      | 7,204            |
| n1g1 | 131072     | 2,000,000  | 16                      | 2                | 137.460     | 12,564           |
| n1g1 | 262144     | 2,000,000  | 16                      | 2                | 283.375     | 23,542           |

#### 1 node 8 devices

| gpu  | batch_size | vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ---------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g8 | 512        | 2,000,000  | 16                      | 2                | 2.295       | 2,304            |
| n1g8 | 1024       | 2,000,000  | 16                      | 2                | 2.557       | 2,316            |
| n1g8 | 2048       | 2,000,000  | 16                      | 2                | 2.866       | 2,338            |
| n1g8 | 4096       | 2,000,000  | 16                      | 2                | 3.536       | 2,388            |
| n1g8 | 8192       | 2,000,000  | 16                      | 2                | 5.174       | 2,488            |
| n1g8 | 16384      | 2,000,000  | 16                      | 2                | 8.499       | 2,688            |
| n1g8 | 32768      | 2,000,000  | 16                      | 2                | 14.248      | 3,080            |
| n1g8 | 65536      | 2,000,000  | 16                      | 2                | 26.879      | 3,804            |
| n1g8 | 131072     | 2,000,000  | 16                      | 2                | 52.243      | 5,226            |
| n1g8 | 262144     | 2,000,000  | 16                      | 2                | 111.140     | 8,314            |

### vocab size X 2 tests
All vocat size double tests were performed with 7 x 1024 Hidden Fully-connected Units.

#### 1 node 1 device

| gpu  | batch_size | vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ---------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g1 | 16384      | 2,000,000  | 16                      | 7                | 61.797      | 5,200            |
| n1g1 | 16384      | 4,000,000  | 16                      | 7                | 63.053      | 5,694            |
| n1g1 | 16384      | 8,000,000  | 16                      | 7                | 65.590      | 6554             |
| n1g1 | 16384      | 16,000,000 | 16                      | 7                | 70.382      | 8402             |
| n1g1 | 16384      | 32,000,000 | 16                      | 7                | 82.165      | 11970            |

#### 1 node 8 devices

| gpu  | batch_size | vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ---------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g8 | 16384      | 2,000,000  | 16                      | 7                | 14.320      | 3,050            |
| n1g8 | 16384      | 4,000,000  | 16                      | 7                | 15.293      | 3,544            |
| n1g8 | 16384      | 8,000,000  | 16                      | 7                | 17.585      | 4,404            |
| n1g8 | 16384      | 16,000,000 | 16                      | 7                | 22.444      | 6252             |
| n1g8 | 16384      | 32,000,000 | 16                      | 7                | 33.562      | 9820             |

### fixed batch size per device tests
All fix batch size per device tests were performed with 7 x 1024 Hidden Fully-connected Units.

| gpu  | batch_size | vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ---------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g8 | 131072     | 2,000,000  | 16                      | 7                | 87.507      | 7,210            |
| n1g4 | 65536      | 2,000,000  | 16                      | 7                | 72.860      | 6,036            |
| n1g2 | 32768      | 2,000,000  | 16                      | 7                | 68.777      | 5,446            |
| n1g1 | 16384      | 2,000,000  | 16                      | 7                | 61.846      | 5,200            |

### fixed total batch size tests
All fixed total batch size tests were performed with 7 x 1024 Hidden Fully-connected Units.

| gpu  | batch_size | vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | ---------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g8 | 16384      | 2,000,000  | 16                      | 7                | 14.209      | 3,046            |
| n1g4 | 16384      | 2,000,000  | 16                      | 7                | 20.468      | 3,142            |
| n1g2 | 16384      | 2,000,000  | 16                      | 7                | 36.347      | 3,738            |
| n1g1 | 16384      | 2,000,000  | 16                      | 7                | 61.874      | 5,200            |