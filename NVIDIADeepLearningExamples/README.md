# NVIDIA DeepLearningExamples 性能测试复现

本目录提供了[NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) 仓库的性能评测复现，目前支持了TensorFlow、MxNet、PyTorch的Resnet50v1.5性能评测 和 TensorFlow、PyTorch的BERT-base的性能评测。



性能评测的物理环境是4台类似 NVIDIA DGX-1 (8x V100 16G)  的机器（配置比DGX-1稍低一些，没有nvlink），软件环境使用的是[NGC 20.03](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)容器。在单机测试结果上近似于NVIDIA的官方公布结果。同时还测试了2机16卡、4机32卡的情况，用于比较各个框架在经过NVIDIA优化之后的横向扩展性。



具体各个框架在各个模型上的性能评测结果、复现方式等信息在本目录下的各个框架的子目录中。
