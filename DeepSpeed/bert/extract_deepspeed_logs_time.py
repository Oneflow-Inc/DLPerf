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

parser = argparse.ArgumentParser(description="flags for benchmark")
parser.add_argument("--log_dir", type=str, default="./logs/deepspeed/bert/bz32", required=True)
parser.add_argument("--output_dir", type=str, default="./result", required=False)
parser.add_argument('--warmup_batches', type=int, default=20)
parser.add_argument('--train_batches', type=int, default=120)
parser.add_argument('--batch_size_per_device', type=int, default=32)

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
    pricition = fname.split("_")[2]
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
    # extract info from file content, e.g. 2020-10-27 11:28:12,892
    pt = re.compile(r"(\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}:\d{1,2},\d{1,3})", re.S)


    s1 = "[timer.py:157:stop] 0/" + str(args.warmup_batches)
    s2 = "[timer.py:157:stop] 0/" + str(args.train_batches)
    start_time = ''
    end_time = ''
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if "SamplesPerSec" in line:
                if s1 in line:
                    start_time = re.findall(pt, line)[0]
                    continue

                if s2 in line:
                    end_time = re.findall(pt, line)[0]
                    t1 = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S,%f")
                    t2 = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S,%f")
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


def compute_median(iter_dict):
    speed_list = [i for i in iter_dict.values()]
    return round(np.median(speed_list), 2)


def compute_average(iter_dict):
    i = 0
    total_speed = 0
    for iter in iter_dict:
        i += 1
        total_speed += iter_dict[iter]
    return round(total_speed / i, 4)


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
    # The iteration output in tensorflow log files is an integer multiple of 10
    assert args.warmup_batches % 10 ==0 and args.train_batches % 10 ==0
    extract_result()
