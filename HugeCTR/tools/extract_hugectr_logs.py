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


def extract_loss_auc_acc(info, log_file):
    print('extract loss, auc and accuarcy from ', log_file)
    with open(log_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if len(ss) < 3:
                continue
            if ss[1] == 'Iter:':
                # [03d12h24m43s][HUGECTR][INFO]: Iter: 1 Time(1 iters): 0.009971s Loss: 0.624022 lr:0.001000
                it = 'Iter' + ss[2]
                info[it] = {'loss': ss[7]}            
            elif ss[2] == 'AUC:':
                # [03d12h24m43s][HUGECTR][INFO]: Evaluation, AUC: 0.484451
                info[it]['auc'] = ss[3].strip()
            elif ss[1] == 'eval_accuracy,':
                # [7485.21, eval_accuracy, 0.484451, 0.002, 1, ]
                info[it]['acc'] = ss[2][:-1]            
    print(info)

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
            info['latency'] = time_accumulate / num_iters           

    with open(mem_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if len(ss) < 5:
                continue
            if ss[0] == 'max':
                info['device0_max_memory_usage'] = int(float(ss[-1].strip()) / 1024 /1024)
            
    print(info)


if __name__ == "__main__":
    logs_list = glob.glob(os.path.join(args.benchmark_log_dir, "*.log"))
    logs_list = sorted(logs_list)
    for log_file in logs_list:
        json_file = log_file[:-3] + 'json'
        info = parse_conf(json_file)

        if info['max_iter'] in [500, 300000]:
            extract_loss_auc_acc(info, log_file)
        else:
            mem_file = log_file[:-3] + 'mem'
            extract_latency(info, args, log_file, mem_file)

