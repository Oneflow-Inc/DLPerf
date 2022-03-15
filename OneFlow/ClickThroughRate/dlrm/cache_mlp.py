import os

# config paths
python_bin = "/path/to/python_bin"
data_dir = "/path/to/dlrm_parquet"
persistent_path = '/path/to/persistent'
script_path = '/path/to/models/RecommenderSystems/dlrm/graph_train.py'

# 
cache_types = [
    ('device_ssd', '16384'),
    ('device_only', '24576'),
    ('device_host', '16384,24576'),
    ('host_ssd', '16384'),
    ('host_only', '24576'),
]
MLP_types = ['FusedMLP', 'MLP']
AMP_types = [False, True]

os.system(f"{python_bin} -m oneflow --doctor")
#column_size_array=[39884407, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532952, 2953546, 403346, 10, 2208, 11938, 155, 4, 976, 14, 39979772, 25641295, 39664985, 585935, 12972, 108, 36]
column_size_array=[145387, 13921, 13816, 6018, 16107, 3, 6653, 1223, 35, 115717, 35480, 37399, 10, 1833, 6066, 58, 4, 842, 14, 154995, 75380, 135108, 29395, 8737, 50, 33]
column_size_array = ','.join([str(i) for i in column_size_array])
num_eval_examples = 89137319
eval_batch_size = 65536
eval_batchs= num_eval_examples // eval_batch_size

emb_size=128
env = f"EMBEDDING_SIZE={emb_size} "
#env += "NCCL_DEBUG=INFO "

dl = f"{python_bin} -m oneflow.distributed.launch "
dl += "--nproc_per_node 8 "
dl += "--nnodes 1 "
dl += "--node_rank 0 "
dl += "--master_addr 127.0.0.1 "
dl += f"{script_path} "

cfg = ""
cfg += "--max_iter 75000 "
cfg += "--loss_print_every_n_iter 1000 "
cfg += "--eval_interval 10000 "
cfg += "--learning_rate 24 "
# cfg += "--model_save_dir ckpt "
# cfg += "--save_model_after_each_eval "
cfg += "--eval_after_training "

cfg += f"--persistent_path {persistent_path} "
cfg += f"--data_dir {data_dir} "
cfg += f"--column_size_array {column_size_array} "
cfg += "--batch_size 55296 "
cfg += f"--eval_batchs {eval_batchs} "
cfg += f"--eval_batch_size {eval_batch_size} "

cfg += f"--bottom_mlp 512,256,{emb_size} "
cfg += "--top_mlp 1024,1024,512,256 "
cfg += f"--embedding_vec_size {emb_size} "
cfg += "--embedding_type OneEmbedding "

i = 0
for cache_type, cache_memory_budget_mb in cache_types:
    for mlp_type in MLP_types:
        for use_fp16 in AMP_types:
            test_name = f"{cache_type}_{mlp_type}"
            var = ""
            var += f"--mlp_type {mlp_type} "
            var += f"--cache_type {cache_type} "
            var += f"--cache_memory_budget_mb {cache_memory_budget_mb} "
            if use_fp16:
                var += "--use_fp16 "
                test_name += "_amp"
            var += f"--test_name {test_name}_{i} "
            cmd = env + dl + cfg + var
            # os.system(cmd)
            print(cmd)
            i += 1
