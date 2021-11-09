import os
import imp
import json
import argparse
from exporter_util import get_metrics_of_node


def get_parser():
    parser = argparse.ArgumentParser("flags for cnn benchmark")

    parser.add_argument("--log_dir", type=str, default="log", help="log directory")
    parser.add_argument("--output", type=str, default="metrics.csv", help="output csv file")
    parser.add_argument("--endswith", type=str, default=".log", help="specify log file extention")
    parser.add_argument("--contains", type=str, default='', help="log filename contains")
    parser.add_argument("--type", type=str, default="cnn", help="cnn, bert")
    parser.add_argument("--start_iter", type=int, default=20)
    parser.add_argument("--end_iter", type=int, default=120)
    parser.add_argument("--master_ip", type=str, default="10.105.0.32",
                        help="master ip address for metric")

    return parser


def meters(result_dict, args):
    needed_keys = [
        args.start_iter,
        args.end_iter,
        'num_nodes',
        'gpu_num_per_node',
        'batch_size_per_device',
    ]
    assert all(key in result_dict.keys() for key in needed_keys)

    duration = float(result_dict[args.end_iter]) - float(result_dict[args.start_iter])

    num_nodes = int(result_dict['num_nodes'])
    gpu_num_per_node = int(result_dict['gpu_num_per_node'])
    batch_size_per_device = int(result_dict['batch_size_per_device'])

    total_batch_size = batch_size_per_device * gpu_num_per_node * num_nodes

    num_batches = args.end_iter - args.start_iter
    num_examples = total_batch_size * num_batches
    throughput = num_examples / duration
    latency = duration / num_batches

    return throughput, latency


def export_csv(all_results, filename):
    assert len(all_results)
    with open(filename, 'w') as f:
        keys = [key for key in list(all_results[0].keys()) if isinstance(key, str)]
        f.write(';'.join(keys) + '\n')
        for res in all_results:
            values = []
            for key in keys:
                values.append(str(res[key]))
            f.write(';'.join(values) + '\n')
    print('save results to', filename)


if __name__ == '__main__':
    FLAGS = get_parser().parse_args()
    if FLAGS.type == 'cnn':
        with open('../Classification/ConvNets/resnet50v1.5/extract_cnn_result.py', 'rb') as fp:
            extract_fn = imp.load_module(
                'extract_info_from_file', fp, 'extract_cnn_result.py',
                ('.py', 'rb', imp.PY_SOURCE)
            ).extract_info_from_file
    elif FLAGS.type == 'bert':
        with open('../LanguageModeling/BERT/extract_bert_result.py', 'rb') as fp:
            extract_fn = imp.load_module(
                'extract_info_from_file', fp, 'extract_bert_result.py',
                ('.py', 'rb', imp.PY_SOURCE)
            ).extract_info_from_file

    all_results = []
    for log_file in os.listdir(FLAGS.log_dir):
        if not log_file.endswith(FLAGS.endswith):
            continue

        if FLAGS.contains not in log_file:
            continue

        res = extract_fn(os.path.join(FLAGS.log_dir, log_file))
        if FLAGS.start_iter not in res or FLAGS.end_iter not in res:
            continue
        res['log_file'] = log_file
        res['throughput'], res['latency'] = meters(res, FLAGS)
        #print(res)
        start = float(res[FLAGS.start_iter]) - 10
        end = float(res[FLAGS.end_iter]) + 5

        # take master node's metric
        node_metrics = get_metrics_of_node(FLAGS.master_ip, start, end)
        for k, v in node_metrics.items():
            if len(v) > 1:
                # take gpu0's metric
                for gpu_metric in v:
                    if gpu_metric['gpu'] == '0':
                        res[k] = gpu_metric['values']
                        break
            elif len(v) == 1:
                res[k] = v[0]['values']
        all_results.append(res)
        #print(json.dumps(node_metrics, indent=4, sort_keys=True))
        #print(json.dumps(res, indent=4, sort_keys=True))
        # print(res)
    export_csv(all_results, FLAGS.output)
