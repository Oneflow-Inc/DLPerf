import os
import sys
import glob
import argparse


parser = argparse.ArgumentParser(description="flags for BERT benchmark")
parser.add_argument(
    "--benchmark_log_dir", type=str, default="./logs/oneflow",
    required=False)
parser.add_argument("--start_iter", type=int, default=19)
parser.add_argument("--end_iter", type=int, default=119)
args = parser.parse_args()


def extract_info_from_file(log_file):
    '''
    batch_size_per_device = 24
    gpu_num_per_node = 1
    num_nodes = 1
    step: 19, total_loss: 9.725, mlm_loss: 8.960, nsp_loss: 0.766, throughput: 54.982 1598056519.8844702
    step: 119, total_loss: 8.010, mlm_loss: 7.331, nsp_loss: 0.679, throughput: 139.896 1598056537.0029895
    '''
    # extract info from file name
    result_dict = {}
    with open(log_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if ss[0] in ['batch_size_per_device', 'gpu_num_per_node', 'num_nodes']:
                result_dict[ss[0]] = ss[2].strip() 
            elif ss[0] == 'step:': 
                it = int(ss[1][:-1]) 
                result_dict[it] = ss[-1].strip()

    return result_dict             


def compute_throughput(result_dict):
    assert args.start_iter in result_dict and args.end_iter in result_dict
    duration = float(result_dict[args.end_iter]) - float(result_dict[args.start_iter])
    
    total_batch_size = int(result_dict['batch_size_per_device']) * \
                       int(result_dict['gpu_num_per_node']) * int(result_dict['num_nodes'])        

    num_examples = total_batch_size * (args.end_iter - args.start_iter)
    throughput = num_examples / duration
    throughput = '{:.1f}'.format(throughput)
    print('|', result_dict['num_nodes'], '|', result_dict['gpu_num_per_node'], '|', result_dict['batch_size_per_device'], '|', throughput, '|')


def extract_result():
    logs_list = glob.glob(os.path.join(args.benchmark_log_dir, "*/*.log"))
    logs_list = sorted(logs_list)
    print('|', 'num_nodes', '|', 'gpu_num_per_node', '|', 'batch_size_per_device', '|', 'throughput', '|')
    print('|', '--------', '|', '--------', '|', '--------', '|', '--------', '|')
    for l in logs_list:
        result_dict = extract_info_from_file(l)
        compute_throughput(result_dict)


if __name__ == "__main__":
    extract_result()

