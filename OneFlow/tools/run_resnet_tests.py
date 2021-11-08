import os
from generate_resnet_test_cmds import init_rn50_tests
from generate_resnet_test_cmds import get_parser, init_rn50_tests

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument("--password", type=str, default="password", help="password for ansible")
    FLAGS = parser.parse_args()

    rn50 = init_rn50_tests(FLAGS)
    cmds = rn50(FLAGS.repeat)
    for num_nodes, cmd in cmds:
        ansible_cmd = ['bash ./ansible_execute.sh']
        ansible_cmd.append(f'--cmd="{cmd}"')
        ansible_cmd.append(f'--num-nodes={num_nodes}')
        ansible_cmd.append(f'--password={FLAGS.password}')
        running_cmd = ' '.join(ansible_cmd)
        print(running_cmd)
        os.system(running_cmd)