import os
import json
import argparse


parser = argparse.ArgumentParser(description="generate hugeCTR config json file base on template.")

parser.add_argument("--template_json", type=str, help="path to template hugectr json file")
parser.add_argument("--output_json", type=str, default='hugectr_conf.json', 
                    help="path to output hugectr json file")
parser.add_argument("--gpu_num_per_node", type=int, default=1)
parser.add_argument('--num_nodes', type=int, default=1, help='node/machine number for training')
parser.add_argument("--total_batch_size", type=int, default=16384)
parser.add_argument("--eval_interval", type=int, default=1000000)#change to 1000(for 300k iters) or 1(500iters) loss auc 
parser.add_argument("--eval_batches", type=int, default=20)
parser.add_argument("--snapshot", type=int, default=1000000)
parser.add_argument("--max_iter", type=int, default=300000)
parser.add_argument("--display", type=int, default=200)

parser.add_argument("--deep_slot_type", type=str, default='Localized', help="Distributed or Localized")
parser.add_argument("--plan_file", type=str, default='all2all_plan_bi_1.json', help="plan file for LocalizedSlot")

parser.add_argument("--wide_vocab_size", type=int, default=2322444)
parser.add_argument("--deep_vocab_size", type=int, default=2322444)
parser.add_argument("--deep_vec_size", type=int, default=16)

args=parser.parse_args()

def parse_conf(conf_file):
    with open(conf_file, "r") as f:
        conf = json.load(f)
    return conf


def get_gpu_conf(args):
    assert args.gpu_num_per_node > 0
    assert args.num_nodes > 0
    node_gpus = [i for i in range(args.gpu_num_per_node)]
    if args.num_nodes > 1:
        return [node_gpus for _ in range(args.num_nodes)] 
    return node_gpus


conf = parse_conf(args.template_json)
conf["solver"]["gpu"] = get_gpu_conf(args)
conf["solver"]["batchsize"] = args.total_batch_size
conf["solver"]["eval_interval"] = args.eval_interval
conf["solver"]["eval_batches"] = args.eval_batches
conf["solver"]["snapshot"] = args.snapshot
conf["solver"]["max_iter"] = args.max_iter
conf["solver"]["display"] = args.display

if args.deep_slot_type.lower == 'distributed':
    sparse_embedding1_type = 'DistributedSlotSparseEmbeddingHash'
    deep_data_type = 'DistributedSlot'
else:
    sparse_embedding1_type = 'LocalizedSlotSparseEmbeddingHash'
    deep_data_type = 'LocalizedSlot'

for layer in conf["layers"]:
    if layer['name'] == 'data':
        for sparse in layer['sparse']:
            if sparse['top'] == 'deep_data':
                sparse['type'] = deep_data_type
    elif layer['name'] == 'sparse_embedding2':#wide_data
        layer['sparse_embedding_hparam']['max_vocabulary_size_per_gpu'] = args.wide_vocab_size
    elif layer['name'] == 'sparse_embedding1':#deep_data
        layer['type'] = sparse_embedding1_type
        if sparse_embedding1_type == 'LocalizedSlotSparseEmbeddingHash':
            layer['plan_file'] = args.plan_file
        layer['sparse_embedding_hparam']['max_vocabulary_size_per_gpu'] = args.deep_vocab_size
        layer['sparse_embedding_hparam']['embedding_vec_size'] = args.deep_vec_size
    elif layer['name'] == 'reshape1':
        layer['leading_dim'] = args.deep_vec_size * 26 

with open(args.output_json, "w") as file:
    json.dump(conf, file, ensure_ascii=True, indent=2, sort_keys=True)

