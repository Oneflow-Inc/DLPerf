import os
import imp
import json
import argparse
from exporter_util import get_metrics_of_node


def get_parser():
    parser = argparse.ArgumentParser("flags for cnn benchmark")

    parser.add_argument("--log_dir", type=str, default="log", help="log directory")
    parser.add_argument("--endswith", type=str, default=".log", help="specify log file extention")
    parser.add_argument("--contains", type=str, default='', help="log filename contains")
    parser.add_argument("--type", type=str, default="cnn", help="cnn, bert")
    parser.add_argument("--start_iter", type=int, default=20)
    parser.add_argument("--end_iter", type=int, default=120)
    parser.add_argument("--master_ip", type=str, default="10.105.0.32",
                        help="master ip address for metric")

    return parser


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

    for log_file in os.listdir(FLAGS.log_dir):
        if not log_file.endswith(FLAGS.endswith):
            continue

        if FLAGS.contains not in log_file:
            continue

        res = extract_fn(os.path.join(FLAGS.log_dir, log_file))
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
        #print(json.dumps(node_metrics, indent=4, sort_keys=True))
        #print(json.dumps(res, indent=4, sort_keys=True))
        print(res)
        