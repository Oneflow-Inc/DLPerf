import os
import argparse


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


if __name__ == "__main__":
    from extract_util import extract_result 
    
    parser = argparse.ArgumentParser(description="flags for BERT benchmark")
    parser.add_argument(
        "--benchmark_log_dir", type=str, default="./logs/oneflow",
        required=False)
    parser.add_argument("--start_iter", type=int, default=19)
    parser.add_argument("--end_iter", type=int, default=119)
    parser.add_argument("--print_mode", type=str, default='markdown')
    args = parser.parse_args()
    extract_result(args, extract_info_from_file)
