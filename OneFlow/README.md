【DLPert】OneFlow Benchmark评测
## Overview
本目录提供一系列脚本用于OneFlow Benchmark评测，评测的主要目标是经典模型的吞吐率，目前评测完成的模型是ResNet50-V1.5和BERT base。

本文提供评测结果的摘要，评测的详细结果报告分别存放于以模型名命名的目录中。 

## OneFlow Benchmark Test Scripts
评测所用脚本都在`scripts`目录中，脚本使用方法参见[scripts/README.md](./scripts/README.md)。

## OneFlow Benchmark Test Result
### ResNet50-V1.5
此处摘取的是2020-08-21的测试结果，详细内容请参考[rn50_fp32_report_0821.md](./ResNet50_v15/rn50_fp32_report_0821.md)
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup | 
| -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 128 | 384.9 | 1.00  | 
| 1 | 8 | 128 | 2969.5 | 7.71  | 
| 2 | 8 | 128 | 5906.6 | 15.35  | 
| 4 | 8 | 128 | 11746.5 | 30.52  | 

### BERT base
此处摘取的是2020-08-22的测试结果，详细内容请参考[bert_base_fp32_report_0822.md](./BERT_base/bert_base_fp32_report_0822.md)
| num_nodes | gpu_num_per_node | batch_size_per_device | throughput | speedup | 
| -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 96 | 149.8 | 1.00  | 
| 1 | 8 | 96 | 1158.5 | 7.73  | 
| 2 | 8 | 96 | 2257.7 | 15.07  | 
| 4 | 8 | 96 | 4456 | 29.75  | 
