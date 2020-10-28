import os
import re
import sys
import glob
import json
import argparse
import pprint
import time
import datetime
import numpy as np

pp = pprint.PrettyPrinter(indent=1)
os.chdir(sys.path[0])

parser = argparse.ArgumentParser(description="flags for cnn benchmark tests data process")
parser.add_argument("-ld", "--log_dir", type=str, default="/workspace/rn50/scripts/ngc/pytorch", required=True)
parser.add_argument("-od", "--output_dir", type=str, default="./result", required=False)
parser.add_argument("-wb", "--warmup_batches", type=int, default=20)
parser.add_argument("-tb", "--train_batches", type=int, default=120)
parser.add_argument("-bz", "--batch_size_per_devic", type=int, default=128)

args = parser.parse_args()


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
        return value


def extract_info_from_file(log_file, result_dict, speed_dict):
    # extract info from file name
    fname = os.path.basename(log_file)
    run_case = log_file.split("/")[-2]  # eg: 1n1g
    model = fname.split("_")[0]
    batch_size = int(fname.split("_")[1].strip("b"))
    precision = fname.split("_")[2]
    test_iter = int(fname.split("_")[3].strip(".log"))
    node_num = int(run_case[0])
    if len(run_case) == 4:
        card_num = int(run_case[-2])
    elif len(run_case) == 5:
        card_num = int(run_case[-3:-1])

    total_batch_size = node_num * card_num * batch_size

    tmp_dict = {
        'average_speed': 0,
        'batch_size_per_device': batch_size,
    }

    avg_speed = 0
    # extract info from file content
    pt = re.compile(r"(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}:\d{1,2}.\d{1,6})", re.S)
    start_time = ''
    end_time = ''
    line_num = 0
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if "Iteration: " in line:
                pt1 = re.compile(r"Iteration: (.*)  train.loss :")
                skip_time = int(re.findall(pt1, line)[0])
                if skip_time > 5:
                    line_num += 1
                    
                    if line_num == args.warmup_batches:
                        start_time = re.findall(pt, line)[0]
                        continue

                    if line_num == args.train_batches:
                        end_time = re.findall(pt, line)[0]
                        t1 = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
                        t2 = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f")
                        cost_time = (t2 - t1).total_seconds()
                        iter_num = args.train_batches - args.warmup_batches
                        avg_speed = round(float(total_batch_size) / (cost_time / iter_num), 2)
                        break

    # compute avg throughoutput
    tmp_dict['average_speed'] = avg_speed
    result_dict[model][run_case]['average_speed'] = avg_speed
    result_dict[model][run_case]['batch_size_per_device'] = tmp_dict['batch_size_per_device']

    speed_dict[model][run_case][test_iter] = avg_speed

    print(log_file, speed_dict[model][run_case])


def compute_speedup(result_dict, speed_dict):
    model_list = [key for key in result_dict]  # eg.['vgg16', 'rn50']
    for m in model_list:
        run_case = [key for key in result_dict[m]]  # eg.['4n8g', '2n8g', '1n8g', '1n4g', '1n1g']
        for d in run_case:
            speed_up = 1.0
            if result_dict[m]['1n1g']['average_speed']:
                result_dict[m][d]['average_speed'] = compute_average(speed_dict[m][d])
                result_dict[m][d]['median_speed'] = compute_median(speed_dict[m][d])
                speed_up = result_dict[m][d]['median_speed'] / compute_median(speed_dict[m]['1n1g'])
            result_dict[m][d]['speedup'] = round(speed_up, 2)

def compute_average(iter_dict):
    i = 0
    total_speed = 0
    for iter in iter_dict:
        i += 1
        total_speed += iter_dict[iter]
    return round(total_speed / i, 2)

def compute_median(iter_dict):
    speed_list = [i for i in iter_dict.values()]
    return round(np.median(speed_list), 2)

def extract_result():
    result_dict = AutoVivification()
    speed_dict = AutoVivification()
    logs_list = glob.glob(os.path.join(args.log_dir, "*/*.log"))
    for l in logs_list:
        extract_info_from_file(l, result_dict, speed_dict)

    # compute speedup
    compute_speedup(result_dict, speed_dict)

    # print result
    pp.pprint(result_dict)

    # write to file as JSON format
    os.makedirs(args.output_dir, exist_ok=True)
    framwork = args.log_dir.split('/')[-1]
    result_file_name = os.path.join(args.output_dir, framwork + "_result.json")
    print("Saving result to {}".format(result_file_name))
    with open(result_file_name, 'w') as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    extract_result()
