# DLPerf - **D**eep**L**earning framework **Perf**ormace profiling toolkit
## Introduction
This repository provides State-of-the-Art classical deeplearning neural network(DNN) models of different deepLearning frameworks that are easy to train and deploy, achieving the best reproducible performance with NVIDIA GPU Server Clusters.

DLPerf measures how fast a deeplearning framework can train DNN models, so both DL framwork and DNN models are involved in this benchmark test.

### Evaluated Deeplearning Framworks
5 deeplearning framworks are evaluated in this repository, they are:
1. OneFlow
2. TensorFlow 1.x and 2.x
3. PyTorch
4. MxNet
5. PaddlePaddle

More and more framworks will be included in the future, such as MindSpore and MegEngine.

### Evaluated Deeplearning Neural Network models
2 classical deeplearning neural network models are tested in this repository, they are:
1. ResNet 50 Version 1.5
2. BERT base

There are a lot of different implementations of these DNN models, we choose official benchmark source as well as [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples). In most cases, we avoid to change any scripts and codes from origin. If we have to, changes were commited by git and mentioned in the documents.

More DNN models will be tested in the future.

### Benchmark Test Scopes
Each DNN model of a framework should be tested on a multi-nodes cluster with different batch size, enable XLA or not, enable auto mixed precision or not.

#### Multi-Nodes and Multi-Devices
We suggest performance each test with 1-node-1-device, 1-node-8-devices, 2-nodes-16-devices, 4-nodes-32-devices. 

#### Batch Size
We talk about batch size, it always means batch size per device during training in this repository. Total batch size will be scaled with total device number for training.

Each DL framework has different device memory management ability, so the maximun batch size per device is vary different with DL frameworks. For this reason, we perform several group test with different batch size.

Normally, higher batch size produces higher performance.

#### XLA 
XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate models with potentially no source code changes. 

We plan to test these DNN models with XLA open and not open if the framework support.  

#### AMP
On some NVIDIA GPUs, Automatic Mixed Precision(AMP) uses FP16 to deliver a performance boost of 3X versus FP32. 

We plan to test these DNN models with AMP open and not open if the framework support.  

### Median Value Principle
According to chapter `Benchmark Test Scopes`, each test case varies with following parameters:
- number of nodes, number of devices
- batch size per device
- XLA 
- AMP

Each test case will repeat several times(suggest 7 times). The median value is choosed as the final result.

### Throughput
Throughput is the average training samples per second, e.g. images/sec for image classification.

To get a continuous and stable output, first several training steps are ignored. In practice, we ignore 20 training steps of the begining, and measure the following 100 steps processing time to calculate throughput.

## Files and Folders
- `README.md`: introduce general information of this repository.
- `NVIDIADeepLearningExamples/`: holds the reproducible scripts and test reports for DNN models from NVIDIA DeepLearningExamples;
- `OneFlow/`: holds the reproducible scripts and test reports for DNN models from OneFlow official benchmark;
- `PaddlePaddle/`: holds the reproducible scripts and test reports for DNN models from PaddlePaddle official benchmark;  
- `reports`: holds rounds of DNN's benchmark test reports.

## Summary of Latest Test Results
This section maintains the summary of the lastest test results. For more and detail information, please find in [reports](./reports) folder.

### ResNet50-V1.5 Training Performance images/sec
Our results were obtained by running the applicable training scripts on 4 server nodes with 8x NVIDIA-V100-16G GPUs each. The specific training script that was run is documented in the corresponding model's README.

The difference between v1 and v1.5 is in the bottleneck blocks which require downsampling. ResNet v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution

This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1, but comes with a small performance drawback (~5% imgs/sec).

| Framework | Source | FP32<br>bsz=128 | FP32 XLA | AMP |
| ---- | ---- | ---- | ---- | --- |
| OneFlow | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) | [11711.2](./OneFlow/ConvNets/rn50_fp32_report_0821.md) | TODO | TODO |
| TensorFlow 1.x | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5) | [9496.91](./NVIDIADeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5) | TODO | TODO |
| MxNet | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) | **QJ** | TODO | TODO |
| PyTorch | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) | **SX** | TODO | TODO |
| PaddlePaddle | [PaddleCV](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification) | [9330.94](./PaddlePaddle/resnet50v1.5) | TODO | TODO |
| TensorFlow 2.x | [TensorFlow-models](https://github.com/tensorflow/models/tree/r2.3.0/official/vision/image_classification) | [9418.44](./TensorFlow/resnet50v1.5) | TODO | TODO |
| MxNet | [**QJ**]() | [**QJ**]() | TODO | TODO |

### BERT base Pretraining Performance sequence/sec
Our results were obtained by running the applicable training scripts on 4 server nodes with 8x NVIDIA-V100-16G GPUs each. The specific training script that was run is documented in the corresponding model's README.

| Framework | Source | FP32<br>bsz=max | FP32<br>bsz=32 |FP32 XLA | AMP |
| ---- | ---- | ---- | ---- | ---- | --- |
| OneFlow | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/LanguageModeling/BERT) | [4456.0<br>bsz=96](./OneFlow/BERT/bert_base_fp32_report_0822.md) | [3715.1](./OneFlow/BERT/bert_base_fp32_report_0822.md) | TODO | TODO |
| TensorFlow 1.x | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT) | [2478.59<br/>bsz=48](./NVIDIADeepLearningExamples/Tensorflow/LanguageModeling/BERT) | [1923.68](./NVIDIADeepLearningExamples/TensorFlow/LanguageModeling/BERT) | TODO | TODO |
| MxNet | [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/v0.10.x/scripts/bert) | **QJ** | **QJ** | TODO | TODO |
| PyTorch | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) | **SX** | **SX** | TODO | TODO |
| PaddlePaddle | [PaddleNLP](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT) | [3167.68<br/>bsz=96](./PaddlePaddle/bert) | [2073.6](./PaddlePaddle/bert) | TODO | TODO |

