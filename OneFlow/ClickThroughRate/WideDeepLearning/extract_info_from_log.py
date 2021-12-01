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

 
def extract_info_from_file(log_file):
    '''
    model = resnet50
    batch_size_per_device = 128
    gpu_num_per_node = 8
    num_nodes = 2
    100 time 1602836839.6445205 loss 0.5471092355251312 
    200 time 1602836842.5530608 loss 0.5126773762702942 
    900 time 1602836863.5008075 loss 0.4933771666884422 
    1000 time 1602836866.4403477 loss 0.49211293876171114 
    1100 time 1602836869.3806155 loss 0.4920539349317551 
    '''
    # extract info from file name
    result_dict = {}
    with open(log_file, 'r') as f:
        first_iter = 0
        first_time = 0
        for line in f.readlines():
            ss = line.split(' ')
            if ss[0] in ['num_nodes', 'gpu_num_per_node', 'batch_size', 'deep_vocab_size','hidden_units_num', 'deep_embedding_vec_size']:
                result_dict[ss[0]] = ss[2].strip() 
            elif len(ss) > 3 and ss[1] == 'time':
                if first_iter == 0:
                    first_iter = int(ss[0].strip())
                    first_time = float(ss[2].strip())
                    result_dict['latency(ms)'] = 0
                elif int(ss[0].strip()) - first_iter == 1000:
                    result_dict['latency(ms)'] = (float(ss[2].strip()) - first_time) 
    mem = extract_mem_info(log_file[:-3] + 'mem')
    result_dict['memory_usage(MB)'] = mem
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flags for OneFlow wide&deep")
    parser.add_argument("--benchmark_log_dir", type=str, required=True)
    args = parser.parse_args()

    logs_list = glob.glob(os.path.join(args.benchmark_log_dir, "*.log"))
    #logs_list = sorted(logs_list)
    chunk_list = {}
    for log_file in logs_list:
        test_result = extract_info_from_file(log_file)
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
        titles = ['log_file', 'gpu', 'batch_size', 'deep_vocab_size','deep_embedding_vec_size', 'hidden_units_num', 'latency(ms)', 'memory_usage(MB)']
        write_line(f, titles, '|', True)
        write_line(f, ['----' for _ in titles], '|', True)
        for result in result_list:
            if 'latency(ms)' not in result.keys():
                print(result['log_file'], 'is not complete!')
                continue
            cells = [value_format(result[title]) for title in titles]
            write_line(f, cells, '|', True)
    
