# OneFlow GPT Test Scripts
  本文介绍如何使用脚本测试OneFlow GPT

### 训练脚本：`scripts/args_pretrain_gpt.sh`
  此脚本需提前配置好项目目录`BASE_DIR=/data/OneFlow-Benchmark/LanguageModeling/GPT`与数据集目录`DATA_PATH=/data/gpt/gpt_sample_dataset_text_document`

- ### 脚本参数
  测试使用的脚本在`scripts`目录下的`args_pretrain_gpt.sh`，可将其移动到[OneFlow-Benchmark]()/LanguageModeling/GPT/examples下。其中确定了`tensor-model-parallel-size`、`pipeline-model-parallel-size`、`NNODES`以及`GPUS_PER_NODE`等参数情况下，即可确定`data_parallel_size`，即`DP*MP*PP=NODES*GPUS_PER_NODE`
  
  - M_P=${1:-1}，`tensor-model-parallel-size`，指定了模型张量被切分到多少个GPU设备上，即`张量模型并行度`
  - P_P=${2:-1}，`pipeline-model-parallel-size`，即`流水模型并行度`，如一个24层的网络，如果`pipeline-model-parallel-size=4`，即表示将24层分为4个stage，每个stage都会由一组GPU设备处理
  - MICRO_BATCH_SIZE=${3:-8}，是每组模型并行的设备使用的`batch size`，如果是纯数据并行，就是每卡`batch size`
  - GLOABAL_BATCH_SIZE=${4:-16}，`micro_batch_size * data_parallel_size = micro_batch_times_data_parallel_size`
  - NNODES=${5:-1}，节点数量
  - GPUS_PER_NODE=${6:-8}，每个节点的GPU数量
  - TRAIN_ITERS=${7:-520}，训练的轮数
  - NUM_LAYERS=${8:-16}，网络模型层数
  - HIDDEN_SIZE=${9:-1536}
  - NUM_ATTENTION_HEADS=${10:-16}
  - SEQ_LENGTH默认为2048

- ### 运行脚本示例
  ```
    bash examples/args_pretrain_gpt.sh 1 4 2 512 4 8 110 32 1536
  ```
