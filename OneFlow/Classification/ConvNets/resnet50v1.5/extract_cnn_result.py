import argparse


def extract_info_from_file(log_file):
    '''
    model = resnet50
    batch_size_per_device = 128
    gpu_num_per_node = 8
    num_nodes = 2
    train: epoch 0, iter 20, loss: 7.087004, top_1: 0.000000, top_k: 0.000000, samples/s: 3988.891 1597933942.9863544
    train: epoch 0, iter 120, loss: 1.050499, top_1: 1.000000, top_k: 1.000000, samples/s: 5917.583 1597933977.6064055
    '''
    # extract info from file name
    result_dict = {}
    with open(log_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if ss[0] in ['model', 'batch_size_per_device', 'gpu_num_per_node', 'num_nodes']:
                result_dict[ss[0]] = ss[2].strip() 
            elif ss[0] == 'train:': 
                it = int(ss[4][:-1]) 
                result_dict[it] = ss[-1].strip()

    return result_dict             


if __name__ == "__main__":
    from extract_util import extract_result 

    parser = argparse.ArgumentParser(description="flags for cnn benchmark")
    parser.add_argument(
        "--benchmark_log_dir", type=str, default="./logs/oneflow",
        required=False)
    parser.add_argument("--start_iter", type=int, default=20)
    parser.add_argument("--end_iter", type=int, default=120)
    parser.add_argument("--print_mode", type=str, default='markdown')
    args = parser.parse_args()
    extract_result(args, extract_info_from_file)
