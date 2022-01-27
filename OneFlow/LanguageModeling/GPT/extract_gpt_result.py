import os
import argparse
from extract_util import extract_result


parser = argparse.ArgumentParser(description="flags for GPT benchmark")
parser.add_argument(
    "--benchmark_log_dir", type=str, default="./logs/oneflow", required=False
)
parser.add_argument("--start_iter", type=int, default=300)
parser.add_argument("--end_iter", type=int, default=400)
parser.add_argument("--print_mode", type=str, default="markdown")
args = parser.parse_args()


def extract_info_from_file(log_file):
    """
    num_nodes ....................................... 1
    num_gpus_per_node ............................... 8
    data_parallel_size .............................. 1
    tensor_model_parallel_size ...................... 8
    pipeline_model_parallel_size .................... 1
    global_batch_size ............................... 32
    micro_batch_size ................................ 32
    num_accumulation_steps .......................... 1
    num_layers ...................................... 16
    hidden_size ..................................... 2304
    num_attention_heads ............................. 16
    seq_length ...................................... 2048
    log_interval .................................... 1
    Training...
    | step     | micro_batches   | samples         | throughput | latency    | loss       |
    | -------- | --------------- | --------------- | ---------- | ---------- | ---------- |
    | 1        | 1               | 32              | 3.65895    | 8.74569    | 11.27187   |
    | 2        | 2               | 64              | 5.92391    | 5.40183    | 22.54614   |
    | 3        | 3               | 96              | 33.08657   | 0.96716    | 33.82825   |
    | 4        | 4               | 128             | 32.91274   | 0.97227    | 45.10602   |
    | 5        | 5               | 160             | 33.05942   | 0.96795    | 56.36795   |
    | 6        | 6               | 192             | 32.97452   | 0.97045    | 67.64371   |
    | 7        | 7               | 224             | 32.75634   | 0.97691    | 78.92993   |
    | 8        | 8               | 256             | 33.13264   | 0.96581    | 90.20315   |
    | 9        | 9               | 288             | 33.01570   | 0.96924    | 101.47802  |
    utilization.gpu [%], memory.used [MiB]
    100 %, 13858 MiB
    100 %, 13994 MiB
    100 %, 13994 MiB
    100 %, 13994 MiB
    100 %, 13994 MiB
    93 %, 13994 MiB
    100 %, 14102 MiB
    100 %, 13850 MiB
    """
    # extract info from file name
    # print('extract file:',log_file)
    result_dict = {}
    with open(log_file, "r") as f:
        for line in f.readlines():
            ss = line.split(" ")
            if len(ss) == 5 and ss[2] in [
                "num_nodes",
                "num_gpus_per_node",
                "data_parallel_size",
                "tensor_model_parallel_size",
                "pipeline_model_parallel_size",
                "micro_batch_size",
                "global_batch_size",
                "num_accumulation_steps",
                "num_layers",
                "hidden_size",
                "num_attention_heads",
                "seq_length",
                "log_interval",
            ]:
                result_dict[ss[2]] = ss[-1].strip()
            elif len(ss) == 4 and "MiB" in line and "utilization" not in line:
                memory_userd = int(ss[-2])
                if (
                    "memory" not in result_dict.keys()
                    or result_dict["memory"] < memory_userd
                ):
                    result_dict["memory"] = memory_userd

            ss = line.split("|")
            if len(ss) == 8 and "loss" not in line and "-" not in line:
                tmp_line = "".join(line.split(" ")).split("|")
                result_dict["throughput_{}".format(tmp_line[1])] = float(tmp_line[4])
                result_dict["latency_{}".format(tmp_line[1])] = (
                    float(tmp_line[5]) * 1000
                )

    return result_dict


if __name__ == "__main__":
    extract_result(args, extract_info_from_file)
