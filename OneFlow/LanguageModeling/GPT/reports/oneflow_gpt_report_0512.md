# OneFlow ResNet50-V1.5 Benchmark Test Report

本报告总结了OneFlow GPT 模型的评测结果。

## Test Environment

所有的测试都是在4台配置了8张 V100-SXM2-16GB GPU的服务器中进行，主要硬软件配置如下：

- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
- OneFlow: 0.3.5+cu112.git.8b222ee
- OneFlow-Benchmark: master
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

## Test Descriptions

- OneFlow版本: [8b222ee](https://github.com/Oneflow-Inc/oneflow/commit/8b222eed25384007e3689913414ab3b1c97a06ea)
- OneFlow Benchmark仓库版本: [master](https://github.com/Oneflow-Inc/OneFlow-Benchmark)

## Test Case

group | case | num-nodes | num-gpus-per-node | data-parallel-size | tensor-model-parallel-size | pipeline-model-parallel-size | micro-batch-size | micro-batch-size-times-data-parallel-size | num-accumulation-steps | global-batch-size | hidden-size | num-attention-heads | num-layers
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
DP |  |  |  |  |  |  |  |  |  |  |  |  | 
 | DP_1x1x1_2_1536x16 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 1536 | 16 | 16
 | DP_8x1x1_16_1536x16 | 1 | 8 | 8 | 1 | 1 | 2 | 16 | 1 | 16 | 1536 | 16 | 16
 | DP_16x1x1_32_1536x16 | 2 | 8 | 16 | 1 | 1 | 2 | 32 | 1 | 32 | 1536 | 16 | 16
 | DP_32x1x1_64_1536x16 | 4 | 8 | 32 | 1 | 1 | 2 | 64 | 1 | 64 | 1536 | 16 | 16
MP |  |  |  |  |  |  |  |  |  |  |  |  | 
 | MP_1x1x1_8_768x12 | 1 | 1 | 1 | 1 | 1 | 8 | 8 | 1 | 8 | 768 | 12 | 12
 | MP_1x8x1_16_1536x16 | 1 | 8 | 1 | 8 | 1 | 16 | 16 | 1 | 16 | 1536 | 16 | 16
 | MP_1x16x1_16_3072x16 | 2 | 8 | 1 | 16 | 1 | 16 | 16 | 1 | 16 | 3072 | 32 | 16
 | MP_1x32x1_16_3072x32 | 4 | 8 | 1 | 32 | 1 | 16 | 16 | 1 | 16 | 3072 | 32 | 32
2D |  |  |  |  |  |  |  |  |  |  |  |  | 
 | 2D_8x1x1_16_1536x16 | 1 | 8 | 8 | 1 | 1 | 2 | 16 | 1 | 16 | 1536 | 16 | 16
 | 2D_1x8x1_16_1536x16 | 1 | 8 | 1 | 8 | 1 | 16 | 16 | 1 | 16 | 1536 | 16 | 16
 | 2D_2x4x1_16_1536x16 | 1 | 8 | 2 | 4 | 1 | 8 | 16 | 1 | 16 | 1536 | 16 | 16
 | 2D_4x2x1_16_1536x16 | 1 | 8 | 4 | 2 | 1 | 4 | 16 | 1 | 16 | 1536 | 16 | 16
 | 2D_2x8x1_16_2304x24 | 2 | 8 | 2 | 8 | 1 | 8 | 16 | 1 | 16 | 2304 | 24 | 24
 | 2D_4x8x1_32_2304x24 | 4 | 8 | 4 | 8 | 1 | 8 | 32 | 1 | 32 | 2304 | 24 | 24
PP |  |  |  |  |  |  |  |  |  |  |  |  | 
 | DP_PP_8x1x4_512_1536x16 | 4 | 8 | 8 | 1 | 4 | 2 | 16 | 32 | 512 | 1536 | 16 | 16
 | MP_PP_1x8x4_512_2304x24 | 4 | 8 | 1 | 8 | 4 | 16 | 16 | 32 | 512 | 2304 | 24 | 24
 | 2D_PP_2x4x4_512_2304x24 | 4 | 8 | 2 | 4 | 4 | 8 | 16 | 32 | 512 | 2304 | 24 | 24
 | 2D_PP_2x8x2_512_2304x24 | 4 | 8 | 2 | 8 | 2 | 8 | 16 | 32 | 512 | 2304 | 24 | 24
 |  |  |  |  |  |  |  |  |  |  |  |  | 

## Test Logs
  所有日志都在`https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/`下。 

group  |  case  |  oneflow_logs
--  |  --  |  --
DP  |    |  
  --  |  DP_1x1x1_2_1536x16  |  perf_test_logs/pretrain_gpt_1n1d_dp1_mp1_pp1_mbz2_gbz2_s2048_l16_h1536_nh16.log
  --  |  DP_8x1x1_16_1536x16  |  perf_test_logs/pretrain_gpt_1n8d_dp8_mp1_pp1_mbz2_gbz16_s2048_l16_h1536_nh16.log
  --  |  DP_16x1x1_32_1536x16  |  perf_test_logs/pretrain_gpt_2n8d_dp16_mp1_pp1_mbz2_gbz32_s2048_l16_h1536_nh16.log
  --  |  DP_32x1x1_64_1536x16  |  perf_test_logs/pretrain_gpt_4n8d_dp32_mp1_pp1_mbz2_gbz64_s2048_l16_h1536_nh16.log
MP  |    |  
 --   |  MP_1x1x1_8_768x12  |  perf_test_logs/pretrain_gpt_1n1d_dp1_mp1_pp1_mbz8_gbz8_s2048_l12_h768_nh12.log
 --   |  MP_1x8x1_16_1536x16  |  perf_test_logs/pretrain_gpt_1n8d_dp1_mp8_pp1_mbz16_gbz16_s2048_l16_h1536_nh16.log
 --   |  MP_1x16x1_16_3072x16  |  perf_test_logs/pretrain_gpt_2n8d_dp1_mp16_pp1_mbz16_gbz16_s2048_l16_h3072_nh32.log
 --   |  MP_1x32x1_16_3072x32  |  perf_test_logs/pretrain_gpt_4n8d_dp1_mp32_pp1_mbz16_gbz16_s2048_l32_h3072_nh32.log
2D  |    |  
 --   |  2D_8x1x1_16_1536x16  |  perf_test_logs/pretrain_gpt_1n8d_dp8_mp1_pp1_mbz2_gbz16_s2048_l16_h1536_nh16.log
 --   |  2D_1x8x1_16_1536x16  |  perf_test_logs/pretrain_gpt_1n8d_dp1_mp8_pp1_mbz16_gbz16_s2048_l16_h1536_nh16.log
 --   |  2D_2x4x1_16_1536x16  |  perf_test_logs/pretrain_gpt_1n8d_dp2_mp4_pp1_mbz8_gbz16_s2048_l16_h1536_nh16.log
 --   |  2D_4x2x1_16_1536x16  |  perf_test_logs/pretrain_gpt_1n8d_dp4_mp2_pp1_mbz4_gbz16_s2048_l16_h1536_nh16.log
 --   |  2D_2x8x1_16_2304x24  |  perf_test_logs/pretrain_gpt_2n8d_dp2_mp8_pp1_mbz8_gbz16_s2048_l24_h2304_nh24.log
 --   |  2D_4x8x1_32_2304x24  |  perf_test_logs/pretrain_gpt_4n8d_dp4_mp8_pp1_mbz8_gbz32_s2048_l24_h2304_nh24.log
PP  |    |  
 --   |  DP_PP_8x1x4_512_1536x16  |  perf_test_logs/pretrain_gpt_4n8d_dp8_mp1_pp4_mbz2_gbz512_s2048_l16_h1536_nh16.log
 --   |  MP_PP_1x8x4_512_2304x24  |  perf_test_logs/pretrain_gpt_4n8d_dp1_mp8_pp4_mbz16_gbz512_s2048_l24_h2304_nh24.log
 --   |  2D_PP_2x4x4_512_2304x24  |  perf_test_logs/pretrain_gpt_4n8d_dp2_mp4_pp4_mbz8_gbz512_s2048_l24_h2304_nh24.log
 --   |  2D_PP_2x8x2_512_2304x24  |  perf_test_logs/pretrain_gpt_4n8d_dp2_mp8_pp2_mbz8_gbz512_s2048_l24_h2304_nh24.log


## Test Result

group  |  case  |  lantency  |  memory  |  throuthput(sample/sec)
--  |  --  |  --  | --  |  -- 
DP  |    |    |    |  
 --   |  DP_1x1x1_2_1536x16  |  406.16 (ms)  |  11066 (MiB)  |  4.92
 --   |  DP_8x1x1_16_1536x16  |  417.952 (ms)  |  12404 (MiB)  |  38.28
 --   |  DP_16x1x1_32_1536x16  |  489.674 (ms)  |  12200 (MiB)  |  65.35
 --   |  DP_32x1x1_64_1536x16  |  518.642 (ms)  |  12208 (MiB)  |  123.4
MP  |    |    |    |  
 --   |  MP_1x1x1_8_768x12  |  525.214 (ms)  |  7416 (MiB)  |  15.23
 --   |  MP_1x8x1_16_1536x16  |  642.134 (ms)  |  6264 (MiB)  |  24.92
 --   |  MP_1x16x1_16_3072x16  |  4819.296 (ms)  |  8678 (MiB)  |  3.32
 --   |  MP_1x32x1_16_3072x32  |  9505.832 (ms)  |  11256 (MiB)  |  1.68
2D  |    |    |    |  
 --   |  2D_8x1x1_16_1536x16  |  417.902 (ms)  |  12396 (MiB)  |  38.29
 --   |  2D_1x8x1_16_1536x16  |  644.4 (ms)  |  6264 (MiB)  |  24.83
 --   |  2D_2x4x1_16_1536x16  |  539.696 (ms)  |  6366 (MiB)  |  29.65
 --   |  2D_4x2x1_16_1536x16  |  543.888 (ms)  |  7368 (MiB)  |  29.42
 --   |  2D_2x8x1_16_2304x24  |  1268.532 (ms)  |  6998 (MiB)  |  12.61
 --   |  2D_4x8x1_32_2304x24  |  1480.34 (ms)  |  6996 (MiB)  |  21.62
PP  |    |    |    |  
 -- |  DP_PP_8x1x4_512_1536x16  |  4006.318 (ms)  |  8712 (MiB)  |  127.8
 -- |  MP_PP_1x8x4_512_2304x24  |  19196.65 (ms)  |  15670 (MiB)  |  26.67
 -- |  2D_PP_2x4x4_512_2304x24  |  13117.188 (ms)  |  11958 (MiB)  |  39.03
 -- |  2D_PP_2x8x2_512_2304x24  |  14375.352 (ms)  |  10708 (MiB)  |  35.62
