import os
import glob
from statistics import median


def compute_throughput(result_dict, args):
    throughput = 0
    latency = 0
    for i in range(args.start_iter,args.end_iter):
        throughput += result_dict['throughput_{}'.format(i)]
        latency += result_dict['latency_{}'.format(i)]


    return latency / (args.end_iter - args.start_iter), throughput / (args.end_iter - args.start_iter)


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

    throughput_final_result_dict = {}
    memory_final_result_dict = {}
    lantency_final_result_dict = {}
    print("## All Results")
    header_list = ['case', 'memory','lantency','throuthput(sample/sec)']
    mode_print(header_list)
    if args.print_mode == 'markdown':
        mode_print(['--------' for _ in range(4)])
    for l in logs_list:
        result_dict = extract_func(l)
        lantency, throughput = compute_throughput(result_dict, args)
        case = "{num_nodes}n{num_gpus_per_node}g_dp{data_parallel_size}_mp{tensor_model_parallel_size}_pp{pipeline_model_parallel_size}_mbs{micro_batch_size}_gbs{global_batch_size}_na{num_accumulation_steps}_l{num_layers}_hs{hidden_size}_nah{num_attention_heads}_sl{seq_length}".format(**result_dict)
        mode_print([case, "{} (MiB)".format(result_dict['memory']), "{} (ms)".format(round(lantency,2)), throughput])

        if case in throughput_final_result_dict:
            throughput_final_result_dict[case].append(throughput)
            memory_final_result_dict[case].append(result_dict['memory'])
            lantency_final_result_dict[case].append(lantency)
        else:
            throughput_final_result_dict[case] = [throughput]
            memory_final_result_dict[case] = [result_dict['memory']]
            lantency_final_result_dict[case] = [lantency]

    # calculate median throughput and speedup
    final_result_list = []
    for k, v in throughput_final_result_dict.items():
        final_result_list.append([k,max(memory_final_result_dict[k]),median(lantency_final_result_dict[k]),median(v)])

    # sort final_result_list
    #final_result_list = sorted(final_result_list, key=lambda x: (-x[2], x[0], x[1]))

    # print results
    print("## Filtered Result `median value`")
    mode_print(['case', 'memory (MiB)','lantency (ms)','throuthput(sample/sec)'])
    if args.print_mode == 'markdown':
        mode_print(['--------' for _ in range(5)])
    for res in final_result_list:
        mode_print(res)

