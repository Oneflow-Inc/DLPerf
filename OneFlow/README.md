# 【DLPerf】OneFlow Benchmark评测

## Overview

本次复现采用了[OneFlow官方仓库](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435)中[ResNet50 v1.5](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/Classification/cnns) 和 [BERT base](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/LanguageModeling/BERT)，目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

目前，该测试覆盖 FP32 及混合精度，后续将持续维护，增加使用其他优化方式的测评。

## OneFlow Benchmark Test Scripts

评测脚本在`scripts`路径下，脚本使用方法参见[scripts/README.md](./scripts/README.md)。

## OneFlow Benchmark Test Results

### ResNet50-V1.5 result on 4 nodes with 8x V100 16G GPUs each

- ### OneFlow v0.2 F16

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP16 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1         | 1                | 256                   | 1472.72                        | 1.00    |
| 1         | 8                | 256                   | 10629.32                       | 7.22    |
| 2         | 8                | 256                   | 17920.40                       | 12.17   |
| 4         | 8                | 256                   | 33141.02                       | 22.50   | 

- ### OneFlow v0.2 F32

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP32 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1         | 1                | 128                   | 397.64                         | 1.00    |
| 1         | 8                | 128                   | 3130.34                        | 7.87    |
| 2         | 8                | 128                   | 6260.30                        | 15.74   |
| 4         | 8                | 128                   | 12411.97                       | 31.21   |

详情参见：[resnet50_oneflow_v0.2_report_1009.md](./ConvNets/resnet50_oneflow_v0.2_report_1009.md)

### BERT base result on 4 nodes with 8x V100 16G GPUs each

- ### OneFlow v0.2 F16 with clip

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP16 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1         | 1                | 160                   | 605.11                         | 1.00    |
| 1         | 8                | 160                   | 4381.66                        | 7.24    |
| 2         | 8                | 160                   | 8075.16                        | 13.34   |
| 4         | 8                | 160                   | 15724.70                       | 25.99   |
| 1         | 1                | 128                   | 589.12                         | 1.00    |
| 1         | 8                | 128                   | 4179.94                        | 7.10    |
| 2         | 8                | 128                   | 7673.42                        | 13.03   |
| 4         | 8                | 128                   | 14729.00                       | 25.00   |
| 1         | 1                | 64                    | 534.72                         | 1.00    |
| 1         | 8                | 64                    | 3399.43                        | 6.36    |
| 2         | 8                | 64                    | 5745.56                        | 10.75   |
| 4         | 8                | 64                    | 9911.78                        | 18.54   |

<br>

- ### OneFlow v0.2 F16 without clip

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP16 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1         | 1                | 160                   | 613.93                         | 1.00    |
| 1         | 8                | 160                   | 4683.36                        | 7.63    |
| 2         | 8                | 160                   | 8777.57                        | 14.30   |
| 4         | 8                | 160                   | 17210.63                       | 28.03   |
| 1         | 1                | 64                    | 552.48                         | 1.00    |
| 1         | 8                | 64                    | 3897.19                        | 7.05    |
| 2         | 8                | 64                    | 6669.93                        | 12.07   |
| 4         | 8                | 64                    | 11195.72                       | 20.26   |

<br>

- ### OneFlow v0.2 F32 with clip

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP32 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1         | 1                | 96                    | 156.02                         | 1.00    |
| 1         | 8                | 96                    | 1201.70                        | 7.70    |
| 2         | 8                | 96                    | 2352.92                        | 15.08   |
| 4         | 8                | 96                    | 4664.10                        | 29.89   |
| 1         | 1                | 64                    | 156.23                         | 1.00    |
| 1         | 8                | 64                    | 1191.13                        | 7.62    |
| 2         | 8                | 64                    | 2297.08                        | 14.70   |
| 4         | 8                | 64                    | 4519.10                        | 28.93   |
| 1         | 1                | 32                    | 152.89                         | 1.00    |
| 1         | 8                | 32                    | 1105.55                        | 7.23    |
| 2         | 8                | 32                    | 2015.78                        | 13.18   |
| 4         | 8                | 32                    | 3689.80                        | 24.13   |

- ### OneFlow v0.2 F32 without clip

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup |
| --------- | ---------------- | --------------------- | ---------- | ------- |
| 1         | 1                | 96                    | 156.25     | 1.00    |
| 1         | 8                | 96                    | 1234.65    | 7.90    |
| 2         | 8                | 96                    | 2425.97    | 15.53   |
| 4         | 8                | 96                    | 4799.64    | 30.72   |
| 1         | 1                | 32                    | 153.94     | 1.00    |
| 1         | 8                | 32                    | 1194.48    | 7.76    |
| 2         | 8                | 32                    | 2181.51    | 14.17   |
| 4         | 8                | 32                    | 4019.45    | 26.11   |

详情参见：[bert_base_oneflow_v0.2_report_1009.md](./BERT/bert_base_oneflow_v0.2_report_1009.md)
