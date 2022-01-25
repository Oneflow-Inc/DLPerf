import argparse
import os
import glob
from statistics import median




def write_line(f, lst, separator=',', start_end=False):
    lst = ['', *lst, ''] if start_end else lst
    f.write(separator.join(lst))
    f.write('\n')


def value_format(value):
    if isinstance(value, float):
        return '{:.3f}'.format(value)
    elif isinstance(value, int):
        return f'{value:,}'
    else:
        return str(value)


def extract_mem_info(mem_file):
    if not os.path.isfile(mem_file):
        return 'NA'

    with open(mem_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if len(ss) < 5:
                continue
            if ss[0] == 'max':
                return int(float(ss[-1].strip()) / 1024 /1024)
    return 'NA'


def extract_info_from_file(log_file, start_iter):
    '''
    batch_size_per_device = 128
    gpu_num_per_node = 8
    num_nodes = 2
    [HUGECTR][02:51:59][INFO][RANK0]: Iter: 100 Time(100 iters): 0.315066s Loss: 0.125152 lr:0.001000
    [HUGECTR][02:51:59][INFO][RANK0]: Iter: 200 Time(100 iters): 0.213347s Loss: 0.106469 lr:0.001000
    ...
    [HUGECTR][02:52:01][INFO][RANK0]: Iter: 1100 Time(100 iters): 0.222373s Loss: 0.100451 lr:0.001000
    max_iter = 1200
    loss_print_every_n_iter = 100
    gpu_num_per_node = 1
    num_nodes = 1
    '''
    # extract info from file name
    result_dict = {}
    loss_print_every_n_iter = 0
    with open(log_file, 'r') as f:
        latencies = []
        vocab_size = []
        for line in f.readlines():
            ss = line.split(' ')
            if ss[0] in ['num_nodes', 'gpu_num_per_node', 'batch_size', 'embedding_vec_size']:
                result_dict[ss[0]] = ss[2].strip()
           # if ss[0] in ['workspace_size_per_gpu_in_mb']:
                # result_dict['vocab_size'] = int((int(ss[2].strip()) * 1024 * 1024 / 4) // 1000000 * 1000000)
            if ss[0] == 'loss_print_every_n_iter':
                loss_print_every_n_iter = float(ss[2].strip())
            elif len(ss) > 3 and ss[1] == 'Iter:' and '[INFO]' in ss[0]:
                if int(ss[2].strip()) != start_iter:
                    latencies.append(float(ss[5].strip()[:-1]))
        if loss_print_every_n_iter > 0:
            result_dict['latency(ms)'] = 1000 * sum(latencies) / len(latencies) / loss_print_every_n_iter
    mem = extract_mem_info(log_file[:-3] + 'mem')
    result_dict['memory_usage(MB)'] = mem
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flags for HugeCTR WDL")
    parser.add_argument("--benchmark_log_dir", type=str, required=True)
    parser.add_argument("--start_iter", type=int, default=1000)
    # parser.add_argument("--end_iter", type=int, default=1100)
    args = parser.parse_args()

    logs_list = sorted(glob.glob(os.path.join(args.benchmark_log_dir, "*.log")), key=os.path.getmtime)
    #logs_list = sorted(logs_list)
    chunk_list = {}
    for log_file in logs_list:
        test_result = extract_info_from_file(log_file, args.start_iter)
        print(test_result)
        json_file = os.path.basename(log_file)[:-4]
        # print(json_file)
        test_result['log_file'] = json_file
        if json_file not in chunk_list.keys():
            chunk_list[json_file] = []
        chunk_list[json_file].append(test_result)
    result_list = []
    for log_name,chunk in chunk_list.items():
        latency_list = []
        for single_result in chunk:
            latency_list.append(single_result['latency(ms)'])
        tmp_chunk = chunk[0]
        tmp_chunk['gpu'] = 'n{}g{}'.format(tmp_chunk['num_nodes'], tmp_chunk['gpu_num_per_node'])
        tmp_chunk['latency(ms)'] = median(latency_list)
        result_list.append(tmp_chunk)
    #with open(os.path.join(args.benchmark_log_dir, 'latency_reprot.md'), 'w') as f:
    report_file = args.benchmark_log_dir + '_latency_report.md'
    with open(report_file, 'w') as f:
        titles = ['log_file', 'gpu', 'batch_size', 'embedding_vec_size', 'latency(ms)', 'memory_usage(MB)']
        write_line(f, titles, '|', True)
        write_line(f, ['----' for _ in titles], '|', True)
        for result in result_list:
            if 'latency(ms)' not in result.keys():
                print(result['log_file'], 'is not complete!')
                continue
            cells = [value_format(result[title]) for title in titles]
            write_line(f, cells, '|', True)
