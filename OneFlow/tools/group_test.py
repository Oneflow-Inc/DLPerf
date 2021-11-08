import os
import re
import copy
import hashlib

class GroupTest(object):
    def __init__(self, name, script, args={}, envs=[], python_bin='python3', log_dir='log'):
        self.name = name
        self.python_bin = python_bin
        self.script = script
        self.envs = envs
        self.log_dir = log_dir

        assert isinstance(args, dict)
        self.args = args

        self.matrix = []
        self.num_of_runs = 0

    def __call__(self, repeat=1):
        assert repeat > 0
        for i in range(repeat):
            self.run_once()

    def run_once(self):
        self.num_of_runs += 1
        prefix = ' '.join(self.envs)
        prefix = prefix + ' ' + self.python_bin
        prefix = prefix + ' ' + self.script

        if len(self.matrix) == 0:
            self.matrix = [{}]
        for args in self.matrix:
            assert isinstance(args, dict)
            running_args = copy.deepcopy(self.args)
            running_args.update(args)

            string_args_list = []
            for key, value in running_args.items():
                s = f'--{key}'
                if value:
                    s += f'={value}'
                string_args_list.append(s)
            string_args = ' '.join(string_args_list)
            cmd = prefix + ' ' + string_args
            log_file = os.path.join(self.log_dir, self.get_log_name(running_args, self.num_of_runs))
            cmd = cmd + ' 2>&1 | tee ' + log_file
            print(cmd)

    def append_matrix(self, args):
        if isinstance(args, dict):
            self.matrix.append(copy.deepcopy(args))
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

        parts.append(os.uname()[1])
        parts.append(str(ext))
        return '_'.join(parts) + '.log'
