# DLPerf - **D**eep **L**earning Framework **Perf**ormace Profiling Toolkit

## Introduction

This repository provides State-of-the-Art classical deep neural network(DNN) models of different deep learning frameworks which are easy to train and deploy, achieving the best reproducible performance with NVIDIA GPU Server Clusters.

DLPerf measures how fast deep learning frameworks can train DNN models, so both DL frameworks and DNN models are involved in this benchmark test.

### Evaluated Deep Learning Frameworks

5 deep learning frameworks are evaluated in this repository, they are:

1. OneFlow
2. TensorFlow 1.x and 2.x
3. PyTorch
4. MxNet
5. PaddlePaddle

More and more frameworks will be included in the future, such as MindSpore and MegEngine.

### Evaluated Deep Neural Network models

2 classical deep neural network models are tested in this repository, they are:

1. ResNet-50 Version 1.5
2. BERT-Base

There are a lot of different implementations of these DNN models, we choose official benchmark source as well as [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples). In most cases, we avoid changing any scripts and codes from origin. If we have to, changes were committed by git and mentioned in the documents.

More DNN models will be tested in the future.

### Benchmark Test Scopes

Each DNN model of a framework should be tested on a multi-node cluster with different batch sizes, XLA enabled or not, auto mixed precision enabled or not.

#### Multi-Node and Multi-Device

We suggest to perform each test with 1-node-1-device, 1-node-8-devices, 2-nodes-16-devices, 4-nodes-32-devices. 

#### Batch Size

We talk about batch size, it always means batch size per device during training in this repository. Total batch size will be scaled with total device number for training.

Each DL framework has different device memory management ability, so the maximum batch size per device is very different with DL frameworks. For this reason, we perform several group tests with different batch sizes.

Normally, higher batch size produces higher performance.

#### XLA 

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate models with potentially no source code changes. 

We plan to test these DNN models with or without XLA if the framework supports.  

#### AMP

On some NVIDIA GPUs, Automatic Mixed Precision(AMP) uses FP16 to deliver a performance boost of 3X versus FP32. 

We plan to test these DNN models with or without AMP if the framework supports.  

### Median Value Principle

According to chapter `Benchmark Test Scopes`, each test case varies with following parameters:

- number of nodes, number of devices
- batch size per device
- XLA 
- AMP

Each test case will repeat several times(suggest 7 times). The median value is chose as the final result.

### Throughput

Throughput is the average training samples per second, e.g. images/sec for image classification.

To get a continuous and stable output, first several training steps are ignored. In practice, we ignore 20 training steps of the beginning, and measure the following 100 steps processing time to calculate throughput.

## Files and Folders

- `README.md`: introduces general information of this repository.
- `NVIDIADeepLearningExamples/`: holds the reproducible scripts and test reports for DNN models from NVIDIA DeepLearningExamples;
- `OneFlow/`: holds the reproducible scripts and test reports for DNN models from OneFlow official benchmark;
- `PaddlePaddle/`: holds the reproducible scripts and test reports for DNN models from PaddlePaddle official benchmark;  
- `TensorFlow/`: holds the reproducible scripts and test reports for DNN models from TensorFlow 2.x official benchmark;
- `PyTorch/`: holds the reproducible scripts and test reports for DNN models from PyTorch official benchmark;
- `MxNet/`: holds the reproducible scripts and test reports for DNN models from [gluon-nlp](https://github.com/dmlc/gluon-nlp) repo;
- `reports`: holds rounds of DNN's benchmark test reports.

## Summary of Latest Test Results

This section maintains the summary of the latest results. For more and more details, please find in [reports](./reports) folder.

### Latest Test Report

[DLPerf Benchmark Test Report v1.0](./reports/dlperf_benchmark_test_report_v1.md) on 4 nodes with 8x Tesla V100-SXM2-16GB GPUs each. 

### ResNet50-V1.5 Training Performance images/sec

Our results were obtained by running the applicable training scripts on 4 nodes with 8x Tesla V100-SXM2-16GB GPUs each. The specific training script that was run is documented in the corresponding model's README. The **bsz** means batch size per GPU.

The difference between v1 and v1.5 is in the bottleneck blocks which require down sampling. ResNet50 v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution

This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1, but comes with a small performance drawback (~5% images/sec).

| Framework      | Source                                                       | FP32 throughput<br>(img/s)  bsz=128                          | FP32 speedup<br>bsz=128 |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- |
| OneFlow        | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) | [11711.18](./OneFlow/ConvNets/rn50_fp32_report_0821.md)      | 30.43                   |
| TensorFlow 1.x | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5) | [9514.64](./NVIDIADeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5) | 26.25                   |
| MxNet          | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/e470c2150abf4179f873cabad23945bbc920cc5f/MxNet/Classification/RN50v1.5) | [10419.21](./NVIDIADeepLearningExamples/MxNet/Classification/RN50v1.5) | 26.74                   |
| PyTorch        | [PyTorch-examples](https://github.com/pytorch/examples/tree/49ec0bd72b85be55579ae8ceb278c66145f593e1) | [10632.33](./PyTorch/resnet50v1.5)                           | 30.00                   |
| PaddlePaddle   | [PaddleCV](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification) | [9348.17](./PaddlePaddle/resnet50v1.5)                       | 26.50                   |
| TensorFlow 2.x | [TensorFlow-models](https://github.com/tensorflow/models/tree/r2.3.0/official/vision/image_classification) | [9418.44](./TensorFlow/resnet50v1.5)                         | 29.27                   |

### BERT base Pretraining Performance sentences/sec

Our results were obtained by running the applicable training scripts on 4 nodes with 8x Tesla V100-SXM2-16GB GPUs each. The specific training script that was run is documented in the corresponding model's README. The **bsz** means batch size per GPU.

| Framework      | Source                                                       | FP32 throughput<br>(sentences/sec), bsz=max                  | FP32 speedup<br>bsz=max | FP32 throughput<br>(sentences/sec), bsz=32                   | FP32 speedup<br/>bsz=32 |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------ | ----------------------- |
| OneFlow        | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/LanguageModeling/BERT) | [4455.97<br>bsz=96](./OneFlow/BERT/bert_base_fp32_report_0822.md) | 29.75                   | [3715.08](./OneFlow/BERT/bert_base_fp32_report_0822.md)      | 25.59                   |
| TensorFlow 1.x | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT) | [2478.59<br/>bsz=48](./NVIDIADeepLearningExamples/TensorFlow/LanguageModeling/BERT) | 22.02                   | [1923.68](./NVIDIADeepLearningExamples/TensorFlow/LanguageModeling/BERT) | 18.01                   |
| MxNet          | [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert) | [1895.50<br/>bsz=32](./MxNet/BERT)                           | 14.92                   | [1895.50](/MxNet/BERT)                                       | 14.92                   |
| PaddlePaddle   | [PaddleNLP](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT) | [3167.68<br/>bsz=96](./PaddlePaddle/bert)                    | 23.13                   | [2073.60](./PaddlePaddle/bert)                               | 15.63                   |

