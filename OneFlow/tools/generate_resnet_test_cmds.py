import os
import re
import argparse
from group_test import GroupTest

class OfResnetTest(GroupTest):
    def __init__(self, name, script, args={}, envs=[], python_bin='python3', log_dir='log',
                 hosts_file='hosts'):
        super(OfResnetTest, self).__init__(name, script, args, envs, python_bin)
        self.update_data_dir()
        self.init_hosts(hosts_file)

    def update_data_dir(self):
        if 'data_dir' in self.args:
            self.args['train_data_dir'] = self.args['data_dir'] + '/train'
            self.args['train_data_part_num'] = 256
            self.args['val_data_dir'] = self.args['data_dir'] + '/validation'
            self.args['val_data_part_num'] = 256
        self.args.pop('data_dir', None)

    def init_hosts(self, host_file):
        pat = re.compile("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
        with open(host_file, 'r') as f:
            lines = f.readlines()
            hosts = []
            for line in lines:
                test_ip = line.strip()
                test = pat.match(test_ip)
                if test:
                    hosts.append(test_ip)
            self.args['node_ips'] = ','.join(hosts)


def init_rn50_tests(FLAGS):
    envs = [
        'PYTHONUNBUFFERED=1',
        'NCCL_LAUNCH_MODE=PARALLEL',
        'ONEFLOW_COMM_NET_IB_ENABLE=1',
    ]

    default_args = {
        'data_dir': '/dataset/ImageNet/ofrecord',
        'optimizer': "sgd",
        'momentum': 0.875,
        'label_smoothing': 0.1,
        'learning_rate': 1.536,
        'loss_print_every_n_iter': 100,
        'val_batch_size_per_device': 50,
        'fuse_bn_relu': True,
        'fuse_bn_add_relu': True,
        'nccl_fusion_threshold_mb': 16,
        'nccl_fusion_max_ops': 24,
        'gpu_image_decoder': True,
        'num_epoch': 1,
        'model': "resnet50"
        #2>&1 | tee ${LOGFILE}
    }

    runs_on = [[1, 1], [1, 8], [2, 8], [4, 8]]

    rn50 = OfResnetTest('resnet50', FLAGS.script, python_bin=FLAGS.python_bin, args=default_args, 
                        envs=envs, log_dir=FLAGS.log_dir)

    num_steps = 1100
    for run_on in runs_on:
        run_on_args = {
            'num_nodes': run_on[0],
            'gpu_num_per_node': run_on[1],
            'channel_last': False,
        }
        num_devices = run_on[0] * run_on[1]

        for bsz in [192, 256]:
            run_on_args['batch_size_per_device'] = bsz
            run_on_args['num_examples'] = bsz * num_devices * num_steps
            rn50.append_matrix(run_on[0], run_on_args)

        run_on_args['use_fp16'] = None
        run_on_args['pad_output'] = None
        run_on_args['channel_last'] = True
        for bsz in [256, 512]:
            run_on_args['batch_size_per_device'] = bsz
            run_on_args['num_examples'] = bsz * num_devices * num_steps
            rn50.append_matrix(run_on[0], run_on_args)

    naming_rule = {
        'num_nodes': 'n',
        'gpu_num_per_node': 'g',
        'batch_size_per_device': 'b',
        'use_fp16': 'amp',
    }
    rn50.set_log_naming_rule(naming_rule)
    return rn50


def get_parser():
    parser = argparse.ArgumentParser("flags for cnn benchmark")

    parser.add_argument("--python_bin", type=str, default="python3", help="python bin path")
    parser.add_argument("--script", type=str, default="Classification/cnns/of_cnn_train_val.py", 
                        help="of_cnn_train_val.py path")
    parser.add_argument("--log_dir", type=str, default="log", help="log directory")
    parser.add_argument("--repeat", type=int, default=1, help="repeat times")
    
    return parser


if __name__ == '__main__':
    FLAGS = get_parser().parse_args()
    rn50 = init_rn50_tests(FLAGS)
    cmds = rn50(FLAGS.repeat)
    for num_nodes, cmd in cmds:
        print(num_nodes, cmd)

