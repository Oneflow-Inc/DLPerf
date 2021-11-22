import os
import re
import copy
import time
import argparse

class GroupTest(object):
    def __init__(self, name, script, args={}, envs=[], python_bin='python3', log_dir='log',
                 hosts_file='hosts', distributed_launch=False):
        self.name = name
        self.python_bin = python_bin
        self.script = script
        self.envs = envs
        self.log_dir = log_dir

        self.distributed_launch = distributed_launch

        assert isinstance(args, dict)
        self.args = args

        self.matrix = []
        self.num_of_runs = 0
        if not distributed_launch:
            self.init_hosts(hosts_file)

    def __call__(self, repeat=1):
        assert repeat > 0
        cmds = []
        for i in range(repeat):
            cmds += self.run_once()
        return cmds

    def run_once(self):
        self.num_of_runs += 1
        prefix = ' '.join(self.envs)
        prefix = prefix + ' ' + self.python_bin
        if self.distributed_launch:
            prefix = 'source set_rank_env.sh; ' + prefix
            prefix = prefix + ' -m oneflow.distributed.launch'

        if len(self.matrix) == 0:
            self.matrix = [{}]

        cmds = []
        for num_nodes, args in self.matrix:
            assert isinstance(args, dict) 
            running_args = copy.deepcopy(self.args)
            running_args.update(args)
            if self.distributed_launch:
                assert 'nproc_per_node' in running_args
                assert 'nnodes' in running_args
                # assert 'node_rank' in dist_args
                assert 'master_addr' in running_args

            log_file = os.path.join(self.log_dir, self.get_log_name(running_args, self.num_of_runs))
            string_args_list = []
            if self.distributed_launch:
                s = '--node_rank=$ONEFLOW_NODE_RANK'
                for key in ['nproc_per_node', 'nnodes', 'master_addr']:
                    s = f'--{key}={running_args[key]}'
                    string_args_list.append(s)
                    running_args.pop(key)

            string_args_list.append(self.script)

            for key, value in running_args.items():
                s = f'--{key}'
                if value:
                    s += f'={value}'
                string_args_list.append(s)
            string_args = ' '.join(string_args_list)
            cmd = prefix + ' ' + string_args
            cmd = cmd + ' 2>&1 | tee ' + log_file
            cmds.append((num_nodes, cmd))
        return cmds

    def append_matrix(self, num_nodes, args):
        if isinstance(args, dict):
            self.matrix.append((num_nodes, copy.deepcopy(args)))
        else:
            assert False

    def set_log_naming_rule(self, rule):
        assert isinstance(rule, dict)
        self.naming_rule = rule

    def get_log_name(self, args, ext):
        #hash_object = hashlib.md5(f'{cmd}{ext}'.encode())
        #return hash_object.hexdigest() + '_' + os.uname()[1] + '.log'
        parts = [self.name]
        for k, v in self.naming_rule.items():
            if k not in args:
                continue

            if args[k]:
                parts.append(v + str(args[k]))
            else:
                parts.append(v)

        # parts.append(os.uname()[1])
        parts.append('`hostname`')
        parts.append(str(ext))
        return '_'.join(parts) + '.log'

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
            self.hosts = hosts
            self.args['node_ips'] = ','.join(hosts)


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
    time.sleep(15)


def exec_cmd(num_nodes, cmd, host_ips, password):
    # Create inventory file
    with open('inventory', 'w') as f:
        for host_ip in host_ips[:num_nodes]:
            f.write(f'{host_ip} ansible_ssh_pass={password}\n')

    with open('tmp_run.sh', 'w') as f:
        f.write(cmd)

    # generate ansible command
    ansible_cmd = ['ansible all --inventory=inventory -m shell',
        # '--ssh-extra-args "-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"',
        f'-a "chdir={os.getcwd()} bash tmp_run.sh"',
    ]
    running_cmd = ' '.join(ansible_cmd)
    print(running_cmd)
    os.system(running_cmd)
    time.sleep(15)


def get_parser():
    parser = argparse.ArgumentParser("flags for oneflow benchmark tests")

    parser.add_argument("--python_bin", type=str, default="python3", help="python bin path")
    parser.add_argument("--script", type=str, default="Classification/cnns/of_cnn_train_val.py", 
                        help="of_cnn_train_val.py path")
    parser.add_argument("--log_dir", type=str, default="log", help="log directory")
    parser.add_argument("--data_dir", type=str, default="./imagenet/ofrecord", help="data directory")
    parser.add_argument("--repeat", type=int, default=1, help="repeat times")
    
    return parser
