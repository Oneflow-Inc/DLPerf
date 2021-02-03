# OneFlow BERT Benchmark Test Report
This folder contains OneFlow BERT Benchmark test reports.  

## Changelog
Note: latest on the top

## Data

- Pretrain datasets
 Note that the dataset is about 200GB, so we provide a [sample dataset](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/wiki_ofrecord_seq_len_128_example.tgz), for testing purposes only.
- SQuAD datasets
Contains the complete dataset and tools. Please click [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/squad_dataset_tools.tgz) for download.
    ```
    squad_dataset_tools
    ├── ofrecord 
    ├── dev-v1.1.json  
    ├── dev-v2.0.json  
    ├── train-v1.1.json  
    ├── train-v2.0.json
    ├── evaluate-v1.1.py  
    ├── evaluate-v2.0.py
    ```
- GLUE(CoLA, MRPC)
Contains the complete dataset. Please click [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/glue_ofrecord.tgz) for download.
    ```
    glue_ofrecord
    ├── CoLA
    │   ├── eval
    │   │   └── eval.of_record-0
    │   ├── test
    │   │   └── predict.of_record-0
    │   └── train
    │       └── train.of_record-0
    └── MRPC
        ├── eval
        │   └── eval.of_record-0
        ├── test
        │   └── predict.of_record-0
        └── train
            └── train.of_record-0
    ```
More information can be found [here](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/LanguageModeling/BERT/README.md).You can also see [here](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/cn/docs/extended_topics/how_to_make_ofdataset.md) how to make the OneFlow dataset.

### OneFlow v0.3.1
- BERT base FP16 with dynamic loss scale test [bert_base_oneflow_v0.3.1_report_1202.md](bert_base_oneflow_v0.3.1_report_1202.md)
### OneFlow v0.2.0
- BERT base without XLA test [bert_base_oneflow_v0.2_report_1009.md](bert_base_oneflow_v0.2_report_1009.md)
### Aug 22nd 2020
- BERT base fp32 without XLA test [report](bert_base_fp32_report_0822.md)
