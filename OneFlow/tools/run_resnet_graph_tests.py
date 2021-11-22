from group_test import GroupTest, exec_cmd, get_parser


def init_tests(FLAGS):
    envs = [
        'PYTHONUNBUFFERED=1',
        'NCCL_LAUNCH_MODE=PARALLEL',
        # 'ONEFLOW_COMM_NET_IB_ENABLE=1',
        'ONEFLOW_DECODER_ENABLE_NVJPEG_HARDWARE_ACCELERATION=0'
    ]

    default_args = {
        # 'master_addr': '10.105.0.54', #default is first ip in hosts file
        'ofrecord-path': FLAGS.data_dir,
        'ofrecord-part-num': 256,
        'lr': 1.536,
        'momentum': 0.875,
        'num-epochs': 1,
        'graph': None,
        'skip-eval': None,
        'print-interval': 10,
        'metric-local': True,
        'metric-train-acc': True,
        'fuse-bn-relu': None,
        'fuse-bn-add-relu': None,
        # 'gpu_image_decoder': True,
        #2>&1 | tee ${LOGFILE}
    }

    runs_on = [[1, 1], [1, 8], [2, 8], [4, 8]]

    rn50 = GroupTest('resnet50_graph', FLAGS.script, python_bin=FLAGS.python_bin, args=default_args, 
                        envs=envs, log_dir=FLAGS.log_dir, distributed_launch=True)

    num_steps = 120
    for run_on in runs_on:
        run_on_args = {
            'nnodes': run_on[0],
            'nproc_per_node': run_on[1],
            'num-devices-per-node': run_on[1],
            # 'channel_last': False,
        }
        num_devices = run_on[0] * run_on[1]

        for bsz in [192, 256]:
            run_on_args['train-batch-size'] = bsz
            run_on_args['samples-per-epoch'] = bsz * num_devices * num_steps
            rn50.append_matrix(run_on[0], run_on_args)

        run_on_args['use-fp16'] = None
        # run_on_args['pad_output'] = None
        # run_on_args['channel_last'] = True
        for bsz in [256, 512]:
            run_on_args['train-batch-size'] = bsz
            run_on_args['samples-per-epoch'] = bsz * num_devices * num_steps
            rn50.append_matrix(run_on[0], run_on_args)

    naming_rule = {
        'nnodes': 'n',
        'nproc_per_node': 'g',
        'train-batch-size': 'b',
        'use-fp16': 'amp',
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
