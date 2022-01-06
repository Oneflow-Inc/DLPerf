# DLPerf OneFlow WideDeepLearning Evaluation
This folder holds OneFlow WideDeepLearning Benchmark Test scripts, tools and reports.

## Folder Structure
```
├── docker
│   ├── build.sh
│   ├── launch.sh
│   └── ubuntu.dockerfile
├── imgs
├── extract_info_from_log.py # extract information from log files
├── extract_info_from_log.sh # bash extract_info_from_log.py
├── gpu_memory_usage.py # log maximum GPU device memory usage during testing
├── README.md
└── scripts
    ├── 300k_iters.sh # 300k iterations test, display loss and auc every 1000 iterations.
    ├── 500_iters.sh # 500 iterations test, display loss and auc every iteration.
    ├── bsz_x2.sh # Batch Size Double Test
    ├── fix_bsz_per_device.sh # test with different number of devices and fixing batch size per device
    ├── fix_total_bsz.sh # test with different number of devices and fixing total batch size
    └── vocab_x2.sh # Vocabulary Size Double Test
```

### Run Scripts
modify all shell files 'WDL_MODEL_DIR'  as path to models/train.py and DATA_DIR as prth to wdl_ofrecord

We can run tests as follows:

1. bash 500iter.sh 
   
   bash 300000iter.sh 
   
   bash bsz_x2.sh
   
   bash vocab_x2.sh
   
   bash fix_total_bsz.sh
   
   bash fix_bsz_per_device.sh
   
2. modify extract_info_from_log.sh 'benchmark_log_dir' as path to log files

   bash extract_info_from_log.sh

## Benchmark Test Cases
This report has summarized OneFlow test on 1 nodes with 8 x Tesla V100 in Dec 2021.

### Test Environment

- 1 nodes with Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz  ($ cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c*)
- Memory 384G ($ cat /proc/meminfo)
- Ubuntu 20.04.3 LTS  ($  cat /etc/issue/) (GNU/Linux 5.4.0-26-generic x86_64)   ($  uname -a)
- CUDA Version: 11.4  ($  nvcc -V), Driver Version: 470.57.02  ($  cat /proc/driver/nvidia/version)
- OneFlow: v0.6.0-3c2e05d49
- models: main@
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
This test will keep batch size per device as a constant value(default is 16384), each test case adopts a different number of GPU devices, such as 1, 2, 4, 8, 16, 32, the total batch size is scaled up with the total number of devices of the test case.

Latency and GPU device memory usage should be recorded in this test.

### Batch Size Double Test
This test uses one GPU device, batch size of the first case is 512 while the subsequent cases are doubled, and so on. This test can be performed on single device, single node and multi-nodes.

Latency and GPU device memory usage should be recorded in this test.

### Vocabulary Size Double Test
This test uses devices as much as possible, the vocabulary size of first case is 3,200,000 while the subsequent cases are doubled, and so on. This test can be performed on single device, single node and multi-nodes.

Latency and GPU device memory usage should be recorded in this test.


## Test Results
We have tested each group for three times, and the median of data was selected as the final result.

If it is a test with multiple devices, the log of device 0 is recorded.

### batch size X 2 tests
#### 1 node 1 device  

| gpu  | batch_size | deep_vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | --------------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g1 | 512        | 2322444         | 16                      | 2                | 1.129       | 1,516            |
| n1g1 | 1024       | 2322444         | 16                      | 2                | 1.503       | 1,528            |
| n1g1 | 2048       | 2322444         | 16                      | 2                | 2.414       | 1,548            |
| n1g1 | 4096       | 2322444         | 16                      | 2                | 4.246       | 1,600            |
| n1g1 | 8192       | 2322444         | 16                      | 2                | 7.609       | 1,674            |
| n1g1 | 16384      | 2322444         | 16                      | 2                | 14.207      | 1,850            |
| n1g1 | 32768      | 2322444         | 16                      | 2                | 26.834      | 2,292            |
| n1g1 | 65536      | 2322444         | 16                      | 2                | 53.775      | 2,890            |
| n1g1 | 131072     | 2322444         | 16                      | 2                | 107.333     | 4,658            |
| n1g1 | 262144     | 2322444         | 16                      | 2                | 223.564     | 8,070            |
| n1g1 | 524288     | 2322444         | 16                      | 2                | 449.482     | 14122            |

#### 1 node 8 devices

| gpu  | batch_size | deep_vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | --------------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g8 | 512        | 2322444         | 16                      | 2                | 2.754       | 1,858            |
| n1g8 | 1024       | 2322444         | 16                      | 2                | 2.720       | 1,862            |
| n1g8 | 2048       | 2322444         | 16                      | 2                | 2.835       | 1,866            |
| n1g8 | 4096       | 2322444         | 16                      | 2                | 3.142       | 1,876            |
| n1g8 | 8192       | 2322444         | 16                      | 2                | 3.699       | 1,906            |
| n1g8 | 16384      | 2322444         | 16                      | 2                | 5.067       | 1,954            |
| n1g8 | 32768      | 2322444         | 16                      | 2                | 7.881       | 2,048            |
| n1g8 | 65536      | 2322444         | 16                      | 2                | 14.322      | 2,246            |
| n1g8 | 131072     | 2322444         | 16                      | 2                | 26.206      | 2,640            |
| n1g8 | 262144     | 2322444         | 16                      | 2                | 50.666      | 3,300            |
| n1g8 | 524288     | 2322444         | 16                      | 2                | 98.287      | 4748             |
| n1g8 | 1048576    | 2322444         | 16                      | 2                | 200.414     | 8022             |
| n1g8 | 2097152    | 2322444         | 16                      | 2                | 424.690     | 13430            |

### vocab size X 2 tests
#### 1 node 1 device  

| gpu  | batch_size | deep_vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | --------------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g1 | 16384      | 3200000         | 16                      | 7                | 56.066      | 2,967            |
| n1g1 | 16384      | 6400000         | 16                      | 7                | 56.603      | 3,749            |
| n1g1 | 16384      | 12800000        | 16                      | 7                | 56.838      | 5,405            |
| n1g1 | 16384      | 25600000        | 16                      | 7                | 56.980      | 8,789            |
| n1g1 | 16384      | 51200000        | 16                      | 7                | 57.000      | 15,429           |

#### 1 node 8 devices

| gpu  | batch_size | deep_vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | --------------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g8 | 16384      | 3200000         | 16                      | 7                | 10.272      | 2,201            |
| n1g8 | 16384      | 6400000         | 16                      | 7                | 10.226      | 2,303            |
| n1g8 | 16384      | 12800000        | 16                      | 7                | 10.260      | 2,509            |
| n1g8 | 16384      | 25600000        | 16                      | 7                | 10.270      | 2,929            |
| n1g8 | 16384      | 51200000        | 16                      | 7                | 10.268      | 3,781            |
| n1g8 | 16384      | 102400000       | 16                      | 7                | 10.256      | 5425             |
| n1g8 | 16384      | 204800000       | 16                      | 7                | 10.230      | 8753             |
| n1g8 | 16384      | 409600000       | 16                      | 7                | 10.277      | 15391            |

### fixed batch size per device tests

| gpu  | batch_size | deep_vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | --------------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g1 | 16384      | 2322444         | 32                      | 7                | 60.405      | 3,347            |
| n1g2 | 32768      | 2322444         | 32                      | 7                | 65.053      | 3,265            |
| n1g4 | 65536      | 2322444         | 32                      | 7                | 65.031      | 3,357            |
| n1g8 | 131072     | 2322444         | 32                      | 7                | 68.407      | 3,873            |

### fixed total batch size tests

| gpu  | batch_size | deep_vocab_size | deep_embedding_vec_size | hidden_units_num | latency(ms) | memory_usage(MB) |
| ---- | ---------- | --------------- | ----------------------- | ---------------- | ----------- | ---------------- |
| n1g1 | 16384      | 2322444         | 32                      | 7                | 59.641      | 3,283            |
| n1g2 | 16384      | 2322444         | 32                      | 7                | 33.309      | 2,511            |
| n1g4 | 16384      | 2322444         | 32                      | 7                | 17.625      | 2,079            |
| n1g8 | 16384      | 2322444         | 32                      | 7                | 10.847      | 2,279            |