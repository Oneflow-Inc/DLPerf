from group_test import GroupTest, exec_cmd, get_parser


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


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument("--password", type=str, default="password", help="password for ansible")
    FLAGS = parser.parse_args()

    bert = init_tests(FLAGS)
    cmds = bert(FLAGS.repeat)
    for num_nodes, cmd in cmds:
        # exec_tmp_run_sh(num_nodes, cmd)
        exec_cmd(num_nodes, cmd, bert.hosts, FLAGS.password)
