import os
import glob
from statistics import median


def compute_throughput(result_dict, args):
    assert args.start_iter in result_dict and args.end_iter in result_dict
    duration = float(result_dict[args.end_iter]) - float(result_dict[args.start_iter])

    num_nodes = int(result_dict['num_nodes'])
    gpu_num_per_node = int(result_dict['gpu_num_per_node'])
    batch_size_per_device = int(result_dict['batch_size_per_device'])

    total_batch_size = batch_size_per_device * gpu_num_per_node * num_nodes

    num_examples = total_batch_size * (args.end_iter - args.start_iter)
    throughput = num_examples / duration

    return num_nodes, gpu_num_per_node, batch_size_per_device, throughput


def get_mode_print(mode):
    def mode_print(lst):
        if mode == 'markdown':
            print('|', ' | '.join(('{:.2f}' if type(v) is float else '{}').format(v) for v in lst), '|')
        else:
            print(','.join(('{:.2f}' if type(v) is float else '{}').format(v) for v in lst))
    return mode_print


def extract_result(args, extract_func):
    mode_print = get_mode_print(args.print_mode)
    logs_list = glob.glob(os.path.join(args.benchmark_log_dir, "*/*.log"))
    logs_list = sorted(logs_list)

    final_result_dict = {}
    print("## All Results")
    mode_print(['num_nodes', 'gpu_num_per_node', 'batch_size_per_device', 'throughput'])
    if args.print_mode == 'markdown':
        mode_print(['--------' for _ in range(4)])
    for l in logs_list:
        result_dict = extract_func(l)
        num_nodes, gpu_num_per_node, batch_size_per_device, throughput = compute_throughput(result_dict, args)
        mode_print([num_nodes, gpu_num_per_node, batch_size_per_device, throughput])
        key = (num_nodes, gpu_num_per_node, batch_size_per_device)
        if key in final_result_dict:
            final_result_dict[key].append(throughput)
        else:
            final_result_dict[key] = [throughput]
    print()

    # calculate n1g1 reference
    n1g1_throughput = {}
    for k, v in final_result_dict.items():
        if k[0] == 1 and k[1] == 1:
            n1g1_throughput[k] = median(v)

    # calculate median throughput and speedup
    final_result_list = []
    for k, v in final_result_dict.items():
        res = list(k)
        res.append(median(v))
        n1g1 = (1, 1, k[2])
        speedup = median(v) / n1g1_throughput[n1g1] if n1g1 in n1g1_throughput else 0.0
        res.append(speedup)
        final_result_list.append(res)

    # sort final_result_list
    final_result_list = sorted(final_result_list, key=lambda x: (-x[2], x[0], x[1]))

    # print results
    print("## Filtered Result `median value`")
    mode_print(['num_nodes', 'gpu_num_per_node', 'batch_size_per_device', 'throughput', 'speedup'])
    if args.print_mode == 'markdown':
        mode_print(['--------' for _ in range(5)])
    for res in final_result_list:
        mode_print(res)

