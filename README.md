# DLPerf - **D**eep**L**earning framework **Perf**ormace profiling toolkit
## Introduction
This repository provides State-of-the-Art classical deeplearning neural network(DNN) models of different deepLearning frameworks that are easy to train and deploy, achieving the best reproducible performance with NVIDIA GPU Server Clusters.

## Computer Vision
### ResNet50-V1.5 Training performance images/sec
Our results were obtained by running the applicable training scripts on 4 server nodes with 8x NVIDIA-V100-16G GPUs each. The specific training script that was run is documented in the corresponding model's README.

| Model | Framework | Source | FP32 | FP32 XLA | AMP |
| ---- | ---- | ---- | ---- | ---- | --- |
| ResNet50-V1.5 | OneFlow | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) | [11711.2](./OneFlow/ConvNets/rn50_fp32_report_0821.md) | TODO | TODO |
| ResNet50-V1.5 | TensorFlow 1.x | [NVIDIA-DeepLearningExamples](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) | [9496.91](./NVIDIADeepLearningExamples/Tensorflow/resnet50v1.5) | TODO | TODO |
| ResNet50-V1.5 | MxNet | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) | **QJ** | TODO | TODO |
| ResNet50-V1.5 | PyTorch | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) | **SX** | TODO | TODO |
| ResNet50-V1.5 | PaddlePaddle | [PaddleCV](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification) | [9330.94](./PaddlePaddle/resnet50v1.5) | TODO | TODO |
| ResNet50-V1.5 | TensorFlow 2.x | [**ZLY**]() | [**ZLY**]() | TODO | TODO |
| ResNet50-V1.5 | MxNet | [**QJ**]() | [**QJ**]() | TODO | TODO |


## Natural Language Processing
### BERT base Pretraining performance sequence/sec
Our results were obtained by running the applicable training scripts on 4 server nodes with 8x NVIDIA-V100-16G GPUs each. The specific training script that was run is documented in the corresponding model's README.

| Model | Framework | Source | FP32 | FP32 XLA | AMP |
| ---- | ---- | ---- | ---- | ---- | --- |
| BERT base Pretrain | OneFlow | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/LanguageModeling/BERT) | [4456.0](https://github.com/Oneflow-Inc/DLPerf/blob/master/OneFlow/BERT/bert_base_fp32_report_0822.md) | TODO | TODO |
| BERT base Pretrain | TensorFlow 1.x | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) | **ZLY** | TODO | TODO |
| BERT base Pretrain | MxNet | [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/v0.10.x/scripts/bert) | **QJ** | TODO | TODO |
| BERT base Pretrain | PyTorch | [NVIDIA-DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) | **SX** | TODO | TODO |
| BERT base Pretrain | PaddlePaddle | [PaddleNLP](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT) | **ZLY** | TODO | TODO |
