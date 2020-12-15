# DLPerf OneFlow WideDeepLearning Evaluation
This folder holds OneFlow WideDeepLearning Benchmark Test scripts, tools and reports.

## Folder Structure
```
├── docker
│   ├── build.sh
│   ├── launch.sh
│   └── ubuntu.dockerfile
├── extract_info_from_log.py # extract information form log files
├── extract_time.py 
├── gpu_memory_usage.py # log maximum GPU device memory usage during testing
├── README.md
├── reports
│   └── wdl_report_1027.md
└── scripts
    ├── 300k_iters.sh # 300k iterations test, display loss and auc every 1000 iterations.

    ├── 500_iters.sh # 500 iterations test, display loss and auc every iteration.

    ├── bsz_x2.sh # Batch Size Double Test
    ├── fix_bsz.sh # test with different number of devices and fixing batch size per device
    ├── local_launch_in_docker.sh # launch oneflow-wdl in docker with arguments
    ├── train_all_in_docker.sh # multi-nodes training scripts
    └── vocab_x2.sh # Vocabulary Size Double Test
```

## Benchmark Test Cases
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
