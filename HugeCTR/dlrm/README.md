# NVIDIA HugeCTR DLRM Benchmark Test 
This folder holds NVIDIA HugeCTR DLRM Benchmark Test scripts, tools and reports.

You can refer to [HugeCTR User Guide](https://github.com/NVIDIA/HugeCTR/blob/master/docs/hugectr_user_guide.md) for additional information.

## folder structure
## Benchmark Test Cases

This report summarized HugeCTR test on 1 nodes with 8 x Tesla V100 on Dec 2021

### Test Environment
- 1 nodes with Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz  ($ cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c*)
- Memory 384G ($ cat /proc/meminfo)
- Ubuntu 20.04.3 LTS  ($  cat /etc/issue/) (GNU/Linux 5.4.0-26-generic x86_64)   ($  uname -a)
- CUDA Version: 11.4  ($  nvcc -V), Driver Version: 470.57.02  ($  cat /proc/driver/nvidia/version)
- HugeCTR version: 3.2
- `nvidia-smi topo -m`

```
		GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	mlx5_0	mlx5_1	CPU Affinity    NUMA Affinity
GPU0	 X 		NV1	    NV2	    NV1	 	SYS		SYS		SYS		NV2		NODE	SYS		0-23,48-71		0
GPU1	NV1	 	X 		NV1		NV2		SYS		SYS		NV2		SYS		NODE	SYS		0-23,48-71		0
GPU2	NV2		NV1	 	X 		NV2		SYS		NV1		SYS		SYS		PIX		SYS		0-23,48-71		0
GPU3	NV1		NV2		NV2		X 		NV1		SYS		SYS		SYS		PIX		SYS		0-23,48-71		0
GPU4	SYS		SYS		SYS		NV1		X 		NV2		NV2		NV1		SYS		NODE	24-47,72-95		1
GPU5	SYS		SYS		NV1		SYS		NV2	 	X 		NV1		NV2		SYS		NODE	24-47,72-95		1
GPU6	SYS		NV2		SYS		SYS		NV2		NV1	 	X 		NV1		SYS		PIX		24-47,72-95		1
GPU7	NV2		SYS		SYS		SYS		NV1		NV2		NV1	 	X 		SYS		PIX		24-47,72-95		1
mlx5_0	NODE	NODE	PIX		PIX		SYS		SYS		SYS		SYS	 	X 		SYS		
mlx5_1	SYS		SYS		SYS		SYS		NODE	NODE	PIX		PIX		SYS	 	X 		

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```



### baseline 

command: python dlrm.py --gpu_num_per_node 4

baseline运行默认参数：

batch_size=65536

learning_rate=0.5     (base learning rate)

warmup_steps=300  (warmup期间 lr = step_ * base_lr_ / warmup_steps_)

decay_start=0

workspace_size_per_gpu_in_mb=11645 

embedding_vec_size=128

max_iter=600

eval_interval=50

### baseline 运行log

[HUGECTR][03:13:26][INFO][RANK0]: Iter: 50 Time(50 iters): 20.430227s Loss: 0.558395 lr:0.085000
[HUGECTR][03:13:43][INFO][RANK0]: Evaluation, AUC: 0.675481
[HUGECTR][03:13:43][INFO][RANK0]: Eval Time for 70 iters: 17.182972s
[HUGECTR][03:14:03][INFO][RANK0]: Iter: 100 Time(50 iters): 37.239699s Loss: 0.536418 lr:0.168333
[HUGECTR][03:14:21][INFO][RANK0]: Evaluation, AUC: 0.695105
[HUGECTR][03:14:21][INFO][RANK0]: Eval Time for 70 iters: 17.082333s
[HUGECTR][03:14:41][INFO][RANK0]: Iter: 150 Time(50 iters): 37.131152s Loss: 0.528158 lr:0.251667
[HUGECTR][03:14:58][INFO][RANK0]: Evaluation, AUC: 0.708745
[HUGECTR][03:14:58][INFO][RANK0]: Eval Time for 70 iters: 17.140160s
[HUGECTR][03:15:18][INFO][RANK0]: Iter: 200 Time(50 iters): 37.168557s Loss: 0.546343 lr:0.335000
[HUGECTR][03:15:35][INFO][RANK0]: Evaluation, AUC: 0.715132
[HUGECTR][03:15:35][INFO][RANK0]: Eval Time for 70 iters: 17.145760s
[HUGECTR][03:15:55][INFO][RANK0]: Iter: 250 Time(50 iters): 37.172421s Loss: 0.534963 lr:0.418333
[HUGECTR][03:16:12][INFO][RANK0]: Evaluation, AUC: 0.720022
[HUGECTR][03:16:12][INFO][RANK0]: Eval Time for 70 iters: 17.117377s
[HUGECTR][03:16:33][INFO][RANK0]: Iter: 300 Time(50 iters): 37.171914s Loss: 0.495738 lr:0.500000
[HUGECTR][03:16:50][INFO][RANK0]: Evaluation, AUC: 0.724995
[HUGECTR][03:16:50][INFO][RANK0]: Eval Time for 70 iters: 17.130679s
[HUGECTR][03:17:10][INFO][RANK0]: Iter: 350 Time(50 iters): 37.130778s Loss: 0.530376 lr:0.500000
[HUGECTR][03:17:27][INFO][RANK0]: Evaluation, AUC: 0.727772
[HUGECTR][03:17:27][INFO][RANK0]: Eval Time for 70 iters: 17.159518s
[HUGECTR][03:17:47][INFO][RANK0]: Iter: 400 Time(50 iters): 37.222825s Loss: 0.526999 lr:0.500000
[HUGECTR][03:18:04][INFO][RANK0]: Evaluation, AUC: 0.728558
[HUGECTR][03:18:04][INFO][RANK0]: Eval Time for 70 iters: 17.187404s
[HUGECTR][03:18:25][INFO][RANK0]: Iter: 450 Time(50 iters): 37.232422s Loss: 0.516090 lr:0.500000
[HUGECTR][03:18:42][INFO][RANK0]: Evaluation, AUC: 0.732136
[HUGECTR][03:18:42][INFO][RANK0]: Eval Time for 70 iters: 17.184398s
[HUGECTR][03:19:02][INFO][RANK0]: Iter: 500 Time(50 iters): 37.203517s Loss: 0.503241 lr:0.500000
[HUGECTR][03:19:19][INFO][RANK0]: Evaluation, AUC: 0.735191
[HUGECTR][03:19:19][INFO][RANK0]: Eval Time for 70 iters: 17.160128s
[HUGECTR][03:19:39][INFO][RANK0]: Iter: 550 Time(50 iters): 37.228689s Loss: 0.504160 lr:0.500000
[HUGECTR][03:19:57][INFO][RANK0]: Evaluation, AUC: 0.737055
[HUGECTR][03:19:57][INFO][RANK0]: Eval Time for 70 iters: 17.186027s

...
[HUGECTR][04:17:04][INFO][RANK0]: Evaluation, AUC: 0.759263
[HUGECTR][04:17:04][INFO][RANK0]: Eval Time for 70 iters: 17.218921s
[HUGECTR][04:17:24][INFO][RANK0]: Finish 1200 iterations with batchsize: 65536 in 879.05s.



### 