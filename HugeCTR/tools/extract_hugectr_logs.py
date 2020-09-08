import os
import glob
import json
import argparse


parser = argparse.ArgumentParser(description="flags for HugeCTR benchmark")
parser.add_argument(
    "--benchmark_log_dir", type=str, default="./logs",
    required=False)
parser.add_argument("--start_iter", type=int, default=100)
parser.add_argument("--end_iter", type=int, default=1100)
parser.add_argument("--print_mode", type=str, default='markdown')
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


def parse_conf(conf_file):
    with open(conf_file, "r") as f:
        conf = json.load(f)
    info = {}
    info['gpu'] = conf["solver"]["gpu"] 
    info['batchsize'] = conf["solver"]["batchsize"] 
    info['max_iter'] = conf["solver"]["max_iter"] 

    for layer in conf["layers"]:
        if layer['name'] == 'sparse_embedding2':#wide_data
            wide_vocab_size = layer['sparse_embedding_hparam']['max_vocabulary_size_per_gpu'] 
        elif layer['name'] == 'sparse_embedding1':#deep_data
            deep_vocab_size = layer['sparse_embedding_hparam']['max_vocabulary_size_per_gpu'] 
            info['deep_vec_size'] = layer['sparse_embedding_hparam']['embedding_vec_size']
    assert wide_vocab_size == deep_vocab_size
    info['vocab_size'] = wide_vocab_size
    return info


def write_line(f, lst, separator=',', start_end=False):
    lst = ['', *lst, ''] if start_end else lst
    f.write(separator.join(lst))
    f.write('\n')

def extract_loss_auc_acc(info, log_file):
    print('extract loss, auc and accuarcy from ', log_file)
    metrics = []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if len(ss) < 3:
                continue
            if ss[1] == 'Iter:':
                # [03d12h24m43s][HUGECTR][INFO]: Iter: 1 Time(1 iters): 0.009971s Loss: 0.624022 lr:0.001000
                # it = {'Iter': ss[2], 'loss': ss[7]}            
                it = [ss[2], ss[7]]            
            elif ss[2] == 'AUC:':
                # [03d12h24m43s][HUGECTR][INFO]: Evaluation, AUC: 0.484451
                # it['auc'] = ss[3].strip()
                it.append(ss[3].strip())
            elif ss[1] == 'eval_accuracy,':
                # [7485.21, eval_accuracy, 0.484451, 0.002, 1, ]
                # it['acc'] = ss[2][:-1]            
                it.append(ss[2][:-1])
                metrics.append(it)
        info['metrics'] = metrics
    #print(info)

    with open(log_file[:-3] + 'csv', 'w') as f:
        write_line(f, ['Iter', 'loss', 'auc', 'acc'])
        for it in info['metrics']:
            write_line(f, it)
        

def extract_latency(info, args, log_file, mem_file):
    print('extract latency from ', log_file)
    with open(log_file, 'r') as f:
        num_iters = 0
        time_accumulate = 0.0
        for line in f.readlines():
            ss = line.split(' ')
            if len(ss) < 7:
                continue
            if ss[1] == 'Iter:':
                # [03d16h06m36s][HUGECTR][INFO]: Iter: 200 Time(100 iters): 2.203441s Loss: 0.475288 lr:0.001000
                it = int(ss[2])
                if it >= args.start_iter and it <= args.end_iter:
                    time_accumulate += float(ss[5][:-1])
                    num_iters += int(ss[3].split('(')[1])
        if num_iters > 0:
            info['latency(ms)'] = time_accumulate / num_iters * 1000 #ms

    with open(mem_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if len(ss) < 5:
                continue
            if ss[0] == 'max':
                info['device0_max_memory_usage(MB)'] = int(float(ss[-1].strip()) / 1024 /1024)
    return info       

def value_format(value):
    if isinstance(value, float):
        return '{:.3f}'.format(value)
    elif isinstance(value, int):
        return f'{value:,}'
    else:
        return str(value)

if __name__ == "__main__":
    logs_list = glob.glob(os.path.join(args.benchmark_log_dir, "*.log"))
    logs_list = sorted(logs_list)
    result_list = []
    for log_file in logs_list:
        json_file = log_file[:-4] + 'json'
        info = parse_conf(json_file)
        info['log_file'] = os.path.basename(log_file)[:-3]

        if info['max_iter'] in [500, 300000]:
            extract_loss_auc_acc(info, log_file)
        else:
            mem_file = log_file[:-3] + 'mem'
            result_list.append(extract_latency(info, args, log_file, mem_file))

    with open(os.path.join(args.benchmark_log_dir, 'latency_reprot.md'), 'w') as f:
        titles = ['log_file', 'gpu', 'batchsize', 'max_iter', 'deep_vec_size', 'vocab_size', 'latency(ms)', 'device0_max_memory_usage(MB)']
        write_line(f, titles, '|', True)
        write_line(f, ['----' for _ in titles], '|', True)
        for result in result_list:
            if 'latency(ms)' not in result.keys():
                print(result['log_file'], 'is not complete!')
                continue
            cells = [value_format(result[title]) for title in titles]
            write_line(f, cells, '|', True)
    
