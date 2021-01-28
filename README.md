# DLPerf - **D**eep **L**earning Framework **Perf**ormance Profiling Toolkit

## Introduction

This repository provides State-of-the-Art classical deep neural network(DNN) models of different deep learning frameworks which are easy to train and deploy, achieving the best reproducible performance with NVIDIA GPU Server Clusters.

DLPerf measures how fast deep learning frameworks can train DNN models, so both DL frameworks and DNN models are involved in this benchmark test.

### Evaluated Deep Learning Frameworks

Multiple deep learning frameworks are evaluated in this repository, they are:

1. OneFlow
2. TensorFlow 1.x and 2.x
3. PyTorch
4. MXNet
5. PaddlePaddle

More frameworks will be included in the future, such as MindSpore, MegEngine, etc.

### Evaluated Deep Neural Network models

There are two main types of model cases tested in this repository, generally including : 

- Common cases

- Special cases


The first type is classical deep neural network models that used to evaluate the performance of each framework,such as:

1. **ResNet-50 v1.5**
2. **BERT-Base**

The secode type is that some models use special techniques or frameworks with unique implementations,such as implementation of [Megatron-LM](https://github.com/microsoft/DeepSpeedExamples/tree/a79272cc8b8f0c5b66c803e581a1355341eacb77/Megatron-LM) based on Microsoft's framwork deepspeed, [HugeCTR](https://github.com/NVIDIA/HugeCTR)(Designed for CTR estimation training and implemented by NVIDIA).

In general, there are a lot of different implementations of these DNN models, we choose official benchmark source as well as [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples). In most cases, we avoid changing any scripts and codes from origin. If we have to, changes were mentioned in the documents.

More DNN models will be tested in the future.

### Benchmark Test Scopes

Each DNN model of a framework should be tested on a multi-node cluster with different batch sizes, XLA enabled or not, auto mixed precision enabled or not.

#### Multi-Node and Multi-Device

We suggest to perform each test with 1-node-1-device, 1-node-8-device, 2-node-16-device, 4-node-32-device configuration.  

#### Batch Size

In this repository, when talking about batch size, it always means the number of samples per device during training. The total batch size is scaled up with the total number of devices for training. 

Because each DL framework has its own device memory management strategy, so the maximum batch size per device is different between DL frameworks. For this reason, we perform several group tests with different batch sizes.

Normally, larger batch size produces better performance.

#### XLA 

[XLA](https://www.tensorflow.org/xla) (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate models with potentially no source code changes. 

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
- `NVIDIADeepLearningExamples/`: holds the reproducible scripts and test reports for DNN models from [NVIDIA DeepLearningExamples]( https://github.com/NVIDIA/DeepLearningExamples),  which includes the frameworks (like TensorFlow 1.x, PyTorch, MXNet) and the corresponding models optimized by NVIDIA;
- `OneFlow/`: holds the reproducible scripts and test reports for DNN models from [OneFlow official benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark);
- `PaddlePaddle/`: holds the reproducible scripts and test reports for DNN models from [PaddlePaddle official benchmark](https://github.com/PaddlePaddle/models/tree/release/1.8);  
- `TensorFlow/`: holds the reproducible scripts and test reports for DNN models from [TensorFlow 2.x official benchmark](https://github.com/tensorflow/models/tree/r2.3.0);
- `PyTorch/`: holds the reproducible scripts and test reports for DNN models from [PyTorch official benchmark](https://github.com/PyTorch/examples/tree/49ec0bd72b85be55579ae8ceb278c66145f593e1);
- `MxNet/`: holds the reproducible scripts and test reports for DNN models from [gluon-nlp](https://github.com/dmlc/gluon-nlp)  and [gluon-cv](https://github.com/dmlc/gluon-cv);
- `reports`: holds rounds of DNN's benchmark test reports.

## Summary of Latest Test Results(common cases)

This section maintains a summary of the results of the common models.For more details, please refer to [reports](./reports) folder.

### Latest Test Report

[DLPerf Benchmark Test Report v1.0](./reports/dlperf_benchmark_test_report_v1.md) on 4 nodes with 8x Tesla V100-SXM2-16GB GPUs each. 

[DLPerf 性能评测报告中文版 v1.0](./reports/dlperf_benchmark_test_report_v1_cn.md)

### ResNet50-V1.5 Training Performance images/sec

Our results were obtained by running the applicable training scripts on 4 nodes with 8x Tesla V100-SXM2-16GB GPUs each. The specific training script that was run is documented in the corresponding model's README. The **bsz** means batch size per GPU.

The difference between v1 and v1.5 is in the bottleneck blocks which require down sampling. ResNet50 v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution

This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1, but comes with a small performance drawback (~5% images/sec).

| Framework                                                    | Source                                                       | FP32 throughput<br>(img/s)  bsz=128                          | FP32 speedup<br>bsz=128 | AMP throughput<br>(img/s) bsz=256                   | AMP speedup<br>bsz=256         |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- | --------------------------------------------------- | ------------------------------ |
| [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.2.0) | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/v0.2.0/Classification/cnns) | [12411.97](./OneFlow)                                        | 31.21                   | 33141.02                                            | 22.50                          |
| [NGC MXNet](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet/tags) | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/e470c2150abf4179f873cabad23945bbc920cc5f/MxNet/Classification/RN50v1.5) | [11233.92](./NVIDIADeepLearningExamples/MxNet/Classification/RN50v1.5) | 28.67                   | 30713.68                                            | 22.03                          |
| [NGC TensorFlow 1.x](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags) | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/Classification/ConvNets/resnet50v1.5) | [9514.64](./NVIDIADeepLearningExamples/TensorFlow/Classification/ConvNets/resnet50v1.5) | 26.25                   | <sup>[1]</sup>29171.69<sup>w/XLA</sup><br>24734.22  | 24.34<sup>w/XLA</sup><br>26.17 |
| [NGC PyTorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags) | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/Classification/ConvNets/resnet50v1.5) | [10917.09](./NVIDIADeepLearningExamples/PyTorch/resnet50v1.5) | 29.72                   | 22551.16                                            | 28.09                          |
| [MXNet](https://github.com/apache/incubator-mxnet/releases/tag/1.6.0) | [gluon-cv](https://github.com/dmlc/gluon-cv/blob/f9a8a284b8/scripts/classification/imagenet/README.md) | [9579.74](./MxNet/Classification/RN50v1b)                    | 24.93                   | 10565.55                                            | 12.67                          |
| [TensorFlow 2.x](https://github.com/tensorflow/tensorflow/tree/v2.3.0) | [TensorFlow-models](https://github.com/tensorflow/models/tree/r2.3.0/official/vision/image_classification) | [9418.44](./TensorFlow/resnet50v1.5)                         | 29.27                   | 19314.31                                            | 17.96                          |
| [PyTorch](https://github.com/pytorch/pytorch/tree/v1.6.0)    | [PyTorch-examples](https://github.com/PyTorch/examples/tree/49ec0bd72b85be55579ae8ceb278c66145f593e1/imagenet) | [10021.29](./PyTorch/resnet50v1.5)                           | 28.75                   | <sup>[2]</sup> -                                    | -                              |
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle/tree/v1.8.3) | [PaddleCV](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification) | [9348.17](./PaddlePaddle/resnet50v1.5)                       | 26.50                   | <sup>[3]</sup>10633.22<br>11617.57<sup>w/DALI</sup> | 10.2<br>13.1<sup>w/DALI</sup>  |

[1]:  AMP throughput of TensorFlow 1.x is obtained **with** or **without** XLA and using **bsz = 224**, because when bsz = 256 OOM (out of memory) will be encountered.

[2]: The PyTorch official benchmark repository [PyTorch-examples](https://github.com/pytorch/examples/tree/49ec0bd72b85be55579ae8ceb278c66145f593e1) does **NOT** support AMP, we will use [NVIDIA-APEX](https://github.com/NVIDIA/apex) plug-in for testing in the near future.

[3]: The AMP throughput 10633.22 img/s of PaddlePaddle is obtained with **bsz = 224** and **without DALI**, because when bsz = 256 OOM will be encountered. The throughput 11617.57 img/s is obtained with **bsz = 196** and **with** [**DALI-paddle plug-in**](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification#dali预处理) because using DALI will occupy more GPU device memory, so bsz = 224 or 256 both encounters OOM. The official data [28594 img/s](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification#混合精度训练) provided by PaddlePaddle is tested on **V100 32G** and the PaddlePaddle docker image with DALI not released, so we cannot replicate this result. If anyone can help us improve PaddlePaddle test results, please contact us by issue.

### BERT base Pretraining Performance sentences/sec

Our results were obtained by running the applicable training scripts on 4 nodes with 8x Tesla V100-SXM2-16GB GPUs each. The specific training script that was run is documented in the corresponding model's README. The **bsz** means batch size per GPU.

| Framework                                                    | Source                                                       | FP32 throughput<br>bsz=max                                   | FP32 throughput<br>bsz=32 | AMP throughput<br>bsz=max         | AMP throughput<br>bsz=64                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------- | --------------------------------- | ------------------------------------------------------------ |
| [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.2.0) | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/v0.2.0/LanguageModeling/BERT) | [4664.10<br>bsz=96](./OneFlow)                               | 3689.80                   | 15724.70<br>bsz=160               | 9911.78                                                      |
| [NGC TensorFlow 1.x](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags) | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/fed7ba99cde958fda12c9e81d12b3d7e738e0590/TensorFlow/LanguageModeling/BERT) | [3089.74<br>bsz=48](./NVIDIADeepLearningExamples/TensorFlow/LanguageModeling/BERT) | 2727.90                   | 11650.0<sup>w/XLA</sup><br>bsz=96 | <sup>[4]</sup>9409.2<sup>w/XLA</sup><br>5189.07<sup>W/O XLA</sup> |
| [NGC PyTorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags) | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/5cc03caa153faab7a2c3b1b5b5d63663f06ce1b4/PyTorch/LanguageModeling/BERT) | [3039.3<br>bsz=48](./NVIDIADeepLearningExamples/PyTorch/BERT) | 2885.81                   | 10349.12<br>bsz=96                | 9331.72                                                      |
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle/tree/v1.8.3) | [PaddleNLP](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT) | [3167.68<br>bsz=96](./PaddlePaddle/bert)                     | 2073.60                   | 5452.35<br>bsz=160                | 3406.36                                                      |
| [OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/v0.2.0)<sup>W/O clip</sup> | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/v0.2.0/LanguageModeling/BERT) | [4799.64<br/>bsz=96](./OneFlow)                              | 4019.45                   | 17210.63<br>bsz=160               | 11195.72                                                     |
| <sup>[5]</sup>[MXNet](https://github.com/apache/incubator-mxnet/tree/1.6.0)<sup>W/O clip</sup> | [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert) | [4340.89<br>bsz=64](./MxNet/BERT)                            | 3671.45                   | 14822.31<br>bsz=128               | 11269.14                                                     |

[4]: AMP throughput of TensorFlow 1.x is obtained **with** or **without** XLA.

[5]: The MXNet BERT script of the [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/7b7bf60259e28b3bf1f4d70569a7e5c18e2f4b3e/scripts/bert) repository does **NOT** support clip_by_ global_norm operation in Adam optimizer. **W/O clip_by_global_norm** operation, the throughput will be larger and the the fine-tuning accuracy may be lower. So we also tested OneFlow data W/O clip operation for comparison.

## Other Test Results(special cases)

This section maintains the results of the special case models.
