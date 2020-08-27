# DLPerf Benchmark Test Report V1.0
This report summrized the results of a series of tests performed on Aug 2020.  

## Test Environment
All tests were performed on 4 GPU Servers with 8x Tesla V100-SXM2-16GB and following is the main hardware and software configurations for each:  
- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)£¬ Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
- `nvidia-smi topo -m`
```
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  CPU Affinity
GPU0     X      NV1     NV1     NV2     NV2     SYS     SYS     SYS     NODE    0-11,24-35
GPU1    NV1      X      NV2     NV1     SYS     NV2     SYS     SYS     NODE    0-11,24-35
GPU2    NV1     NV2      X      NV2     SYS     SYS     NV1     SYS     PIX     0-11,24-35
GPU3    NV2     NV1     NV2      X      SYS     SYS     SYS     NV1     PIX     0-11,24-35
GPU4    NV2     SYS     SYS     SYS      X      NV1     NV1     NV2     SYS     12-23,36-47
GPU5    SYS     NV2     SYS     SYS     NV1      X      NV2     NV1     SYS     12-23,36-47
GPU6    SYS     SYS     NV1     SYS     NV1     NV2      X      NV2     SYS     12-23,36-47
GPU7    SYS     SYS     SYS     NV1     NV2     NV1     NV2      X      SYS     12-23,36-47
mlx5_0  NODE    NODE    PIX     PIX     SYS     SYS     SYS     SYS      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

```

## Evaluated DNN models 
### ResNet50 V1.5
| Framework Version | Docker From | DNN modeles source | Abbr |
| ---- | ---- | ---- | ---- |
| OneFlow 0.1.9 | NA | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) | OF_rn50 |
| TensorFlow 1.15.2 | nvcr.io/nvidia/tensorflow:20.03-tf1-py3 | [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets) | TF_NGC_rn50 |
| MxNet 1.6.0 | nvcr.io/nvidia/mxnet:20.03-py3 | [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) | MX_NGC_rn50 |
| PyTorch | nvcr.io/nvidia/pytorch:20.03-py3 | [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets) | PT_NGC_rn50 |
| PaddlePaddle 1.8.3.post107 | NA | [PaddleCV](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification) | PP_rn50 |

### BERT Base Pretrain 
| Framework Version | Docker From | DNN modeles source | Abbr |
| ---- | ---- | ---- | ---- |
| OneFlow 0.1.9 | NA | [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) | OF_bert |
| TensorFlow 1.15.2 | nvcr.io/nvidia/tensorflow:20.03-tf1-py3 | [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) | TF_NGC_bert |
| MxNet 1.6.0 | nvcr.io/nvidia/mxnet:20.03-py3 | [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/v0.10.x/scripts/bert) | MX_bert |
| PyTorch | nvcr.io/nvidia/pytorch:20.03-py3 | [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) | PT_NGC_bert |
| PaddlePaddle 1.8.3.post107  | NA | [PaddleCV](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification) | PP_rn50 |

## Benchmark Test Options
- Devices: 1 node(1 device), 1 node(8 devices), 2 nodes(8 devices), 4 nodes(8 devices)   
- DataType: Float32
- XLA, TensorRT, AMP: Not Apply

## Benchmark Test Results
### ResNet50 V1.5
data source:
- OF_rn50: [OneFlow ResNet50-V1.5 Benchmark Test Report](../OneFlow/ConvNets/rn50_fp32_report_0821.md)
- TF_NGC_rn50: [NVIDIA-Tensorflow-ResNet50V1.5²âÆÀ](../NVIDIADeepLearningExamples/Tensorflow/Classification/ConvNets/resnet50v1.5)
- MX_NGC_rn50: TODO
- PT_NGC_rn50: TODO
- PP_50: [¡¾DLPerf¡¿Paddle-ResNet50V1.5²âÆÀ](../PaddlePaddle/resnet50v1.5)
#### batch size = 128
| node num | device num | throughput<br>OF_rn50 | throughput<br>TF_NGC_rn50 | throughput<br>MX_NGC_rn50 | throughput<br>PT_NGC_rn50 | throughput<br>PP_rn50 | 
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 384.90 | 362.44 | TODO | TODO | 352.72 | 
| 1 | 8 | 2969.50 | 2721.98 | TODO | TODO | 2625.38 | 
| 2 | 16 | 5906.60 | 5099.42 | TODO | TODO | 4895.27 | 
| 4 | 32 | 11711.20 | 9514.64 | TODO | TODO | 9348.17 | 

### BERT Base Pretrain 
data source:
- OF_rn50: [OneFlow BERT Benchmark Test Report](../OneFlow/BERT/bert_base_fp32_report_0822.md)
- TF_NGC_rn50: [¡¾DLPerf¡¿NVIDIA-Tensorflow-BERT²âÆÀ](../NVIDIADeepLearningExamples/Tensorflow/LanguageModeling/BERT)
- MX_NGC_rn50: TODO
- PT_NGC_rn50: TODO
- PP_50: [¡¾DLPerf¡¿Paddle-BERT²âÆÀ](../PaddlePaddle/bert)

#### batch size = 96
| node num | device num | throughput<br>OF_bert | throughput<br>TF_NGC_bert | throughput<br>MX_bert | throughput<br>PT_NGC_bert | throughput<br>PP_bert |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 1 | 1 | 149.8 | OOM | TODO | TODO | 136.97 |
| 1 | 8 | 1158.5 | OOM | TODO | TODO | 868.6 |
| 2 | 16 | 2257.7 | OOM | TODO | TODO | 1631.36 |
| 4 | 32 | 4456 | OOM | TODO | TODO | 3167.68 |

#### batch size = 64 
| node num | device num | throughput<br>OF_bert | throughput<br>TF_NGC_bert | throughput<br>MX_bert | throughput<br>PT_NGC_bert | throughput<br>PP_bert |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 1 | 1 | 149.8 | OOM | TODO | TODO | 137.27 |
| 1 | 8 | 1138.9 | OOM | TODO | TODO | 761.22 |
| 2 | 16 | 2189.3 | OOM | TODO | TODO | 1426.52 |
| 4 | 32 | 4310.4 | OOM | TODO | TODO | 2736.78 |

#### batch size = 32
| node num | device num | throughput<br>OF_bert | throughput<br>TF_NGC_bert | throughput<br>MX_bert | throughput<br>PT_NGC_bert | throughput<br>PP_bert | 
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 145.2 | 106.8 | TODO | TODO | 132.64 | 
| 1 | 8 | 1043 | 806.56 | TODO | TODO | 615.12 | 
| 2 | 16 | 1890.3 | 1090.2 | TODO | TODO | 1116.02 | 
| 4 | 32 | 3715.1 | 1923.68 | TODO | TODO | 2073.6 |

