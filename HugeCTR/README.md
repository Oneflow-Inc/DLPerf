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