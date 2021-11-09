from generate_bert_test_cmds import get_parser, init_tests
from group_test import exec_cmd


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument("--password", type=str, default="password", help="password for ansible")
    FLAGS = parser.parse_args()

    bert = init_tests(FLAGS)
    cmds = bert(FLAGS.repeat)
    for num_nodes, cmd in cmds:
        # exec_tmp_run_sh(num_nodes, cmd)
        exec_cmd(num_nodes, cmd, bert.hosts, FLAGS.password)

