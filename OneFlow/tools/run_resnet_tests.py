from group_test import GroupTest, exec_cmd, get_parser

class OfResnetTest(GroupTest):
    def __init__(self, name, script, args={}, envs=[], python_bin='python3', log_dir='log',
                 hosts_file='hosts'):
        super(OfResnetTest, self).__init__(name, script, args, envs, python_bin, log_dir, hosts_file)
        self.update_data_dir()

    def update_data_dir(self):
        if 'data_dir' in self.args:
            self.args['train_data_dir'] = self.args['data_dir'] + '/train'
            self.args['train_data_part_num'] = 256
            # self.args['val_data_dir'] = self.args['data_dir'] + '/validation'
            # self.args['val_data_part_num'] = 256
        self.args.pop('data_dir', None)


def init_tests(FLAGS):
    envs = [
        'PYTHONUNBUFFERED=1',
        'NCCL_LAUNCH_MODE=PARALLEL',
        # 'ONEFLOW_COMM_NET_IB_ENABLE=1',
        'ONEFLOW_DECODER_ENABLE_NVJPEG_HARDWARE_ACCELERATION=0'
    ]

    default_args = {
        'data_dir': FLAGS.data_dir,
        'optimizer': "sgd",
        'momentum': 0.875,
        'label_smoothing': 0.1,
        'learning_rate': 1.536,
        'loss_print_every_n_iter': 10,
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

    num_steps = 120
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



if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument("--password", type=str, default="password", help="password for ansible")
    FLAGS = parser.parse_args()

    rn50 = init_tests(FLAGS)
    cmds = rn50(FLAGS.repeat)
    for num_nodes, cmd in cmds:
        # exec_tmp_run_sh(num_nodes, cmd)
        exec_cmd(num_nodes, cmd, rn50.hosts, FLAGS.password)
