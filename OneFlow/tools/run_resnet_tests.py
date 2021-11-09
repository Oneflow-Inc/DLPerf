import os
import time
from generate_resnet_test_cmds import init_rn50_tests
from generate_resnet_test_cmds import get_parser, init_rn50_tests


def exec_tmp_run_sh(num_nodes, cmd):
    ansible_cmd = ['bash ./ansible_execute.sh']
    with open('tmp_run.sh', 'w') as f:
        f.write(cmd)
    ansible_cmd.append(f'--cmd="bash tmp_run.sh"')
    ansible_cmd.append(f'--num-nodes={num_nodes}')
    ansible_cmd.append(f'--password={FLAGS.password}')
    running_cmd = ' '.join(ansible_cmd)
    print(running_cmd)
    os.system(running_cmd)
    time.sleep(3)


def exec_cmd(num_nodes, cmd, host_ips, password):
    # Create inventory file
    with open('inventory', 'w') as f:
        for host_ip in host_ips[:num_nodes]:
            f.write(f'{host_ip} ansible_ssh_pass={password}\n')

    with open('tmp_run.sh', 'w') as f:
        f.write(cmd)

    # generate ansible command
    ansible_cmd = ['ansible all --inventory=inventory -m shell',
        '--ssh-extra-args "-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"',
        f'-a "chdir={os.getcwd()} bash tmp_run.sh"',
    ]
    running_cmd = ' '.join(ansible_cmd)
    print(running_cmd)
    os.system(running_cmd)
    time.sleep(3)

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument("--password", type=str, default="password", help="password for ansible")
    FLAGS = parser.parse_args()

    rn50 = init_rn50_tests(FLAGS)
    cmds = rn50(FLAGS.repeat)
    for num_nodes, cmd in cmds:
        # exec_tmp_run_sh(num_nodes, cmd)
        exec_cmd(num_nodes, cmd, rn50.hosts, FLAGS.password)

