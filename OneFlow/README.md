# 【DLPerf】OneFlow Benchmark评测

## Overview

本次复现采用了[OneFlow官方仓库](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435)中[ResNet50 v1.5](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/Classification/cnns) 和 [BERT base](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/637bb9cdb4cc1582f13bcc171acbc8a8089d9435/LanguageModeling/BERT)，目的在于速度测评，同时根据测速结果给出1机、2机器、4机情况下的加速比，评判框架在分布式多机训练情况下的横向拓展能力。

本文提供评测结果的摘要，评测的详细结果报告分别存放于以模型名命名的目录中。 

目前，该测试仅覆盖 FP32 精度，后续将持续维护，增加混合精度训练，XLA 等多种方式的测评。

<br>

## OneFlow Benchmark Test Scripts

评测脚本在`scripts`路径下，脚本使用方法参见[scripts/README.md](./scripts/README.md)。

<br>

## OneFlow Benchmark Test Results

### ResNet50-V1.5 result on 4 nodes with 8x V100 16G GPUs each

 <br>

1. 此处摘取的是OneFlow v0.2的F16测试结果，详细内容请参考[oneflow_v0.2_rn50_fp16_report.md](./ConvNets/oneflow_v0.2_rn50_fp16_report.md)

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP16 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1 | 1 | 256 | 1472.72 | 1.00 |
| 1 | 8 | 256 | 10629.32 | 7.22 |
| 2 | 8 | 256 | 17920.40 | 12.17 |
| 4 | 8 | 256 | 33141.02 | 22.50 | 

<br>

2. 此处摘取的是OneFlow v0.2的F32测试结果，详细内容请参考[oneflow_v0.2_rn50_fp32_report.md](./ConvNets/oneflow_v0.2_rn50_fp32_report.md)


| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP32 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1 | 1 | 128 | 397.64 | 1.00 |
| 1 | 8 | 128 | 3130.34 | 7.87 |
| 2 | 8 | 128 | 6260.30 | 15.74 |
| 4 | 8 | 128 | 12411.97 | 31.21 |

<br>

### BERT base result on 4 nodes with 8x V100 16G GPUs each

<br>

1. 此处摘取的是OneFlow v0.2的F16的测试结果，详细内容请参考[bert_base_oneflow_v0.2_fp16_report.md](./BERT/bert_base_oneflow_v0.2_fp16_report.md)

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP16 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1 | 1 | 160 | 605.11 | 1.00 |
| 1 | 8 | 160 | 4381.66 | 7.24 |
| 2 | 8 | 160 | 8075.16 | 13.34 |
| 4 | 8 | 160 | 15724.70 | 25.99 |
| 1 | 1 | 128 | 589.12 | 1.00 |
| 1 | 8 | 128 | 4179.94 | 7.10 |
| 2 | 8 | 128 | 7673.42 | 13.03 |
| 4 | 8 | 128 | 14729.00 | 25.00 |
| 1 | 1 | 64 | 534.72 | 1.00 |
| 1 | 8 | 64 | 3399.43 | 6.36 |
| 2 | 8 | 64 | 5745.56 | 10.75 |
| 4 | 8 | 64 | 9911.78 | 18.54 |

<br>

2. 此处摘取的是OneFlow v0.2的F32的测试结果，详细内容请参考[bert_base_oneflow_v0.2_fp32_report.md](./BERT/bert_base_oneflow_v0.2_fp32_report.md)

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP32 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1 | 1 | 96 | 156.02 | 1.00 |
| 1 | 8 | 96 | 1201.70 | 7.70 |
| 2 | 8 | 96 | 2352.92 | 15.08 |
| 4 | 8 | 96 | 4664.10 | 29.89 |
| 1 | 1 | 64 | 156.23 | 1.00 |
| 1 | 8 | 64 | 1191.13 | 7.62 |
| 2 | 8 | 64 | 2297.08 | 14.70 |
| 4 | 8 | 64 | 4519.10 | 28.93 |
| 1 | 1 | 32 | 152.89 | 1.00 |
| 1 | 8 | 32 | 1105.55 | 7.23 |
| 2 | 8 | 32 | 2015.78 | 13.18 |
| 4 | 8 | 32 | 3689.80 | 24.13 |

<br>

3. 此处摘取的是OneFlow v0.2的F16在`clip_gradient=None`情况下的测试结果，详细内容请参考[bert_base_oneflow_v0.2_fp16_no_clip_report.md](./BERT/bert_base_oneflow_v0.2_fp16_no_clip_report.md)

| num_nodes | gpu_num_per_node | batch_size_per_device | throughput <br>FP16 <br>no XLA | speedup |
| --------- | ---------------- | --------------------- | ------------------------------ | ------- |
| 1 | 1 | 64 | 552.48 | 1.00 |
| 1 | 8 | 64 | 3897.19 | 7.05 |
| 2 | 8 | 64 | 6669.93 | 12.07 |
| 4 | 8 | 64 | 11195.72 | 20.26 |
