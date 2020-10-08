# 【DLPerf】OneFlow Benchmark评测

## Overview

本次复现采用了[OneFlow官方仓库](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435)中[ResNet50 v1.5](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/Classification/cnns) 和 [BERT base](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/LanguageModeling/BERT)，目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

本文提供评测结果的摘要，评测的详细结果报告分别存放于以模型名命名的目录中。 

目前，该测试仅覆盖 FP32 精度，后续将持续维护，增加混合精度训练，XLA 等多种方式的测评。

## OneFlow Benchmark Test Scripts

评测脚本在`scripts`路径下，脚本使用方法参见[scripts/README.md](./scripts/README.md)。

## OneFlow Benchmark Test Results

### ResNet50-V1.5 result on 4 nodes with 8x V100 16G GPUs each

此处摘取的是OneFlow v0.2的F16测试结果，详细内容请参考[oneflow_v0.2_rn50_fp16_report.md](./ConvNets/oneflow_v0.2_rn50_fp16_report.md)

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP32 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1 | 1 | 256 | 1472.72 | 1.00 |
| 1 | 8 | 256 | 10629.32 | 7.22 |
| 2 | 8 | 256 | 17920.40 | 12.17 |
| 4 | 8 | 256 | 33141.02 | 22.50 |

此处摘取的是OneFlow v0.2的F32测试结果，详细内容请参考[oneflow_v0.2_rn50_fp32_report.md](./ConvNets/oneflow_v0.2_rn50_fp32_report.md)

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP32 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1 | 1 | 128 | 397.64 | 1.00 |
| 1 | 8 | 128 | 3130.34 | 7.87 |
| 2 | 8 | 128 | 6260.30 | 15.74 |
| 4 | 8 | 128 | 12411.97 | 31.21 |

### BERT base result on 4 nodes with 8x V100 16G GPUs each

此处摘取的是2020-08-22的测试结果，详细内容请参考[bert_base_fp32_report_0822.md](./BERT/bert_base_fp32_report_0822.md)

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP32 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1         | 1                | 96                    | 149.84                         | 1.00    |
| 1         | 8                | 96                    | 1158.51                        | 7.73    |
| 2         | 8                | 96                    | 2257.71                        | 15.07   |
| 4         | 8                | 96                    | 4455.97                        | 29.75   |

