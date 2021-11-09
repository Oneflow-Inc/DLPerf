import argparse
from group_test import GroupTest

class OfResnetTest(GroupTest):
    def __init__(self, name, script, args={}, envs=[], python_bin='python3', log_dir='log',
                 hosts_file='hosts'):
        super(OfResnetTest, self).__init__(name, script, args, envs, python_bin, hosts_file)


def init_tests(FLAGS):
    envs = [
        'PYTHONUNBUFFERED=1',
        'NCCL_LAUNCH_MODE=PARALLEL',
        # 'ONEFLOW_COMM_NET_IB_ENABLE=1',
        # 'ONEFLOW_DECODER_ENABLE_NVJPEG_HARDWARE_ACCELERATION=0'
    ]

    default_args = {
        'data_dir': FLAGS.data_dir,
        'data_part_num': 64,
        'learning_rate': 1e-4,
        'iter_num': 140,
        'loss_print_every_n_iter': 20,
        'seq_length': 128,
        'max_predictions_per_seq': 20,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'max_position_embeddings': 512,
        # 'nccl_fusion_threshold_mb': 16,
        # 'nccl_fusion_max_ops': 24,
        'type_vocab_size': 2,
        'vocab_size': 30522,
        'attention_probs_dropout_prob': 0.1,
        'hidden_dropout_prob': 0.1,
        'model_save_every_n_iter': 10000,
        # 'save_last_snapshot': None,
        'model_save_dir': "output",
    }

    runs_on = [[1, 1], [1, 8], [2, 8], [4, 8]]

    bert = GroupTest('bert', FLAGS.script, python_bin=FLAGS.python_bin, args=default_args, 
                     envs=envs, log_dir=FLAGS.log_dir)

    for run_on in runs_on:
        run_on_args = {
            'num_nodes': run_on[0],
            'gpu_num_per_node': run_on[1],
        }
        num_devices = run_on[0] * run_on[1]

        for bsz in [128, 192]:
            run_on_args['batch_size_per_device'] = bsz
            bert.append_matrix(run_on[0], run_on_args)

        run_on_args['use_fp16'] = None
        for bsz in [256, 384]:
            run_on_args['batch_size_per_device'] = bsz
            bert.append_matrix(run_on[0], run_on_args)

    naming_rule = {
        'num_nodes': 'n',
        'gpu_num_per_node': 'g',
        'batch_size_per_device': 'b',
        'use_fp16': 'amp',
    }
    bert.set_log_naming_rule(naming_rule)
    return bert


def get_parser():
    parser = argparse.ArgumentParser("flags for cnn benchmark")

    parser.add_argument("--python_bin", type=str, default="python3", help="python bin path")
    parser.add_argument("--script", type=str, default="LanguageModeling/BERT/run_pretraining.py", 
                        help="run_pretraining.py path")
    parser.add_argument("--log_dir", type=str, default="log", help="log directory")
    parser.add_argument("--data_dir", type=str, default="./wiki_seq_len_128", help="data directory")
    parser.add_argument("--repeat", type=int, default=1, help="repeat times")
    
    return parser


if __name__ == '__main__':
    FLAGS = get_parser().parse_args()
    bert = init_tests(FLAGS)
    cmds = bert(FLAGS.repeat)
    for num_nodes, cmd in cmds:
        print(num_nodes, cmd)

