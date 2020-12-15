import argparse
from extract_info_from_log import extract_info_from_file

parser = argparse.ArgumentParser(description="flags for OneFlow wide&deep")
parser.add_argument("--log_file", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    result_dict = extract_info_from_file(args.log_file)
    print(result_dict['latency(ms)'])

