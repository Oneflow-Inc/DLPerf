import os
import imp
import argparse


def get_parser():
    parser = argparse.ArgumentParser("flags for cnn benchmark")

    parser.add_argument("--log_dir", type=str, default="log", help="log directory")
    parser.add_argument("--endswith", type=str, default=".log", help="specify log file extention")
    parser.add_argument("--contains", type=str, default='', help="log filename contains")
    parser.add_argument("--type", type=str, default="cnn", help="cnn, bert")

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
        print(res)