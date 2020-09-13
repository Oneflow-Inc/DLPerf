# NVIDIA HugeCTR Benchmark Test 
This folder holds NVIDIA HugeCTR Benchmark Test scripts, tools and reports.

## folder structure
```
DLPerf/HugeCTR $ tree
.
├── docker
│   ├── build.sh
│   ├── Dockerfile
│   └── launch.sh
├── README.md
├── reports
├── scripts
│   ├── 300k_iters.sh # 300k iterations test, display loss and auc every 1000 iterations.
│   ├── 500_iters.sh # 500 iterations test, display loss and auc every iteration.
│   ├── bsz_x2_run.sh # increasing batch size test
│   ├── fix_bsz_per_device_run.sh # test with different number of devices and fixing batch size per device
│   ├── fix_total_bsz_run.sh # test with different number of devices and fixing total batch size 
│   ├── vocab_x2_1node_run.sh # increasing vocab size test
│   ├── multi_node_run.sh # multi nodes test, TODO
│   ├── gpu_memory_usage.py # log maximum GPU device memory usage during testing
│   ├── gen_hugectr_conf_json.py # generate HugeCTR conf json file for testing
│   ├── wdl_2x1024.json # template file for HugeCTR conf json generator, 2x1024 units FCs 
│   └── wdl_7x1024.json # template file for HugeCTR conf json generator, 7x1024 units FCs 
└── tools
    └── extract_hugectr_logs.py
```
## Benchmark Test Cases
### 500 Iterations Test
This test aims to show the training loss convergency profile and validation area under the ROC Curve(AUC) during 500 steps testing. 

In order to show the tendency of the loss and auc curves clealy, the total training batch size is set to 512, 512 is a small value comparison to WDL industry production scenario. Validation should be performed every training step.

### 300,000 Iterations Test
This test aims to show the training loss profile and validation area under the ROC Curve(AUC) during 300,000 steps testing. Compare to 500 iters test, we print loss and auc every 1000 steps, it will bring us a long period view of loss and auc curves.

### Fixed Total Batch Size Test
This test will keep total batch size as a constant value(default is 16384), each test case adopts different number of GPU devices, such as 1, 2, 4, 8, 16, 32.

Latency and GPU device memory usage is recored in this test.

### Fixed Batch Size per Device Test
This test will keep batch size per device as a constant value(default is 16384), each test case adopts different number of GPU devices, such as 1, 2, 4, 8, 16, 32, the total batch size is scaled up with the total number of devices of the test case.

Latency and GPU device memory usage is recored in this test.

### Batch Size Double Test
This test uses one GPU device, the first case's batch size is 512, the batch size of the subsequent case is doubled and so on.

Latency and GPU device memory usage is recored in this test.

### Vocabulary Size Double Test
This test uses devices as much as possible, the first case's vocabulary size is 3,200,000, the vocabsize size of the subsequent case is doubled and so on.

Latency and GPU device memory usage is recored in this test.
