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

def extract_info_from_file_for_models(log_file):
    '''
[rank:0] iter: 100/1200, loss: 0.0831875279545784, latency(ms): 81.5818255022168159 | 2021-12-01 13:19:02.625
[rank:0] iter: 200/1200, loss: 0.0780148208141327, latency(ms): 2.2327776625752449 | 2021-12-01 13:19:02.848
...
[rank:0] iter: 1200/1200, loss: 0.0711858719587326, latency(ms): 2.3108293302357197 | 2021-12-01 13:19:05.145
    '''
    # extract info from file name
    result_dict = {}
    with open(log_file, 'r') as f:
        latencies = []
        for line in f.readlines():
            ss = line.strip().split(' ')
            if ss[0] in ['num_nodes',  'batch_size', 'batch_size_per_proc', 'vocab_size','embedding_vec_size']:
                result_dict[ss[0]] = ss[2].strip() 
            elif len(ss) > 6 and ss[1] == 'iter:' and ss[3] == 'loss:':
                latencies.append(float(ss[6].strip()))
        
        result_dict['gpu_num_per_node'] = int(int(result_dict['batch_size']) / int(result_dict['batch_size_per_proc']))
        result_dict['num_nodes'] = 1

        if len(latencies) > 2:
            latencies.pop(0)
            latencies.pop(-1)

        if len(latencies) > 0:
            result_dict['latency(ms)'] = sum(latencies) / len(latencies)
        else:
            result_dict['latency(ms)'] = 'NA'

    mem = extract_mem_info(log_file[:-3] + 'mem')
    result_dict['memory_usage(MB)'] = mem
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flags for OneFlow wide&deep")
    parser.add_argument("--benchmark_log_dir", type=str, required=True)
    parser.add_argument("--repo", type=str, default='benchmark', help='benchmark or models')
    args = parser.parse_args()

    logs_list = sorted(glob.glob(os.path.join(args.benchmark_log_dir, "*.log")), key=os.path.getmtime)
    #logs_list = sorted(logs_list)
    chunk_list = {}
    for log_file in logs_list:
        if args.repo == 'benchmark':
            test_result = extract_info_from_file(log_file)
        else:
            test_result = extract_info_from_file_for_models(log_file)

        print(test_result)
        json_file = os.path.basename(log_file)[:-4]
        # json_file = os.path.basename(log_file)[:-13]
        print(json_file)
        test_result['log_file'] = json_file
        if json_file not in chunk_list.keys():
            chunk_list[json_file] = []
        chunk_list[json_file].append(test_result)
    result_list = []
    for log_name, chunk in chunk_list.items():
        latency_list = []
        for single_result in chunk:
            if 'latency(ms)' in single_result:
                latency_list.append(single_result['latency(ms)'])
        tmp_chunk = chunk[0]
        tmp_chunk['gpu'] = 'n{}g{}'.format(tmp_chunk['num_nodes'], tmp_chunk['gpu_num_per_node'])
        if len(latency_list):
            tmp_chunk['latency(ms)'] = median(latency_list)
            result_list.append(tmp_chunk)
        else:
            print('latency is not calculated in ', log_name)
    #with open(os.path.join(args.benchmark_log_dir, 'latency_reprot.md'), 'w') as f:
    report_file = args.benchmark_log_dir + '_latency_report.md'
    with open(report_file, 'w') as f:
        titles = ['log_file', 'gpu', 'batch_size', 'vocab_size','embedding_vec_size',  'latency(ms)', 'memory_usage(MB)']
        write_line(f, titles, '|', True)
        write_line(f, ['----' for _ in titles], '|', True)
        for result in result_list:
            if 'latency(ms)' not in result.keys():
                print(result['log_file'], 'is not complete!')
                continue
            cells = [value_format(result[title]) for title in titles]
            write_line(f, cells, '|', True)
