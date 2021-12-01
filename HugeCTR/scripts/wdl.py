import hugectr
from mpi4py import MPI

def WideAndDeep(args):
    vvgpu = [[g for g in range(args.gpu_num_per_node)] for n in range(args.num_nodes)]
    solver = hugectr.CreateSolver(max_eval_batches = args.eval_batchs,
                                batchsize_eval = args.batch_size,
                                batchsize = args.batch_size,
                                lr = args.learning_rate,
                                vvgpu = vvgpu,
                                repeat_dataset = True,
                                i64_input_key = True)
    reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                    source = [f"{args.data_dir}/train/_file_list.txt"],
                                    eval_source = f"{args.data_dir}/val/_file_list.txt",
                                    slot_size_array = [225945, 354813, 202260, 18767, 14108, 6886, 18578, 4, 6348, 1247, 51, 186454, 71251, 66813, 11, 2155, 7419, 60, 4, 922, 15, 202365, 143093, 198446, 61069, 9069, 74, 34],
                                    check_type = hugectr.Check_t.Non)
    optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                        update_type = hugectr.Update_t.Global,
                                        beta1 = 0.9,
                                        beta2 = 0.999,
                                        epsilon = 0.0000001)
    model = hugectr.Model(solver, reader, optimizer)
    model.add(hugectr.Input(label_dim = 1, label_name = "label",
                            dense_dim = args.num_dense_fields, dense_name = "dense",
                            data_reader_sparse_param_array =
                            [hugectr.DataReaderSparseParam("wide_data", 1, True, args.num_wide_sparse_fields),
                            hugectr.DataReaderSparseParam("deep_data", 2, False, args.num_deep_sparse_fields)]))
    model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
                                workspace_size_per_gpu_in_mb = 8,
                                embedding_vec_size = 1,
                                combiner = "sum",
                                sparse_embedding_name = "sparse_embedding2",
                                bottom_name = "wide_data",
                                optimizer = optimizer))
    model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
                                workspace_size_per_gpu_in_mb = 114,
                                embedding_vec_size = args.deep_embedding_vec_size,
                                combiner = "sum",
                                sparse_embedding_name = "sparse_embedding1",
                                bottom_name = "deep_data",
                                optimizer = optimizer))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                                bottom_names = ["sparse_embedding1"],
                                top_names = ["reshape1"],
                                leading_dim=416))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                                bottom_names = ["sparse_embedding2"],
                                top_names = ["reshape2"],
                                leading_dim=2))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReduceSum,
                                bottom_names = ["reshape2"],
                                top_names = ["wide_redn"],
                                axis = 1))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                                bottom_names = ["reshape1", "dense"],
                                top_names = ["concat1"]))
    bottom_name = "concat1"
    for h in range(1, args.hidden_units_num + 1):
        model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                    bottom_names = [bottom_name],
                                    top_names = [f"fc{h}"],
                                    num_output=args.hidden_size))
        model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                    bottom_names = [f"fc{h}"],
                                    top_names = [f"relu{h}"]))
        model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                                    bottom_names = [f"relu{h}"],
                                    top_names = [f"dropout{h}"],
                                    dropout_rate=args.deep_dropout_rate))
        bottom_name = f"dropout{h}"

    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = [bottom_name],
                                top_names = [f"fc{args.hidden_units_num + 1}"],
                                num_output=1))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,
                                bottom_names = [f"fc{args.hidden_units_num + 1}", "wide_redn"],
                                top_names = ["add1"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                                bottom_names = ["add1", "label"],
                                top_names = ["loss"]))
    return model


def get_args(print_args=True):
    import argparse
    def str_list(x):
        return x.split(',')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_format', type=str, default='ofrecord', help='ofrecord or onerec')
    parser.add_argument(
        "--use_single_dataloader_thread",
        action="store_true",
        help="use single dataloader threads per node or not."
    )
    parser.add_argument('--data_dir', type=str, default='/dataset/d4f7e679/criteo_day_0_parquet')
    parser.add_argument('--eval_batchs', type=int, default=300)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--wide_vocab_size', type=int, default=3200000)
    parser.add_argument('--deep_vocab_size', type=int, default=3200000)
    parser.add_argument('--deep_embedding_vec_size', type=int, default=16)
    parser.add_argument('--deep_dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_dense_fields', type=int, default=13)
    parser.add_argument('--num_wide_sparse_fields', type=int, default=2)
    parser.add_argument('--num_deep_sparse_fields', type=int, default=26)
    parser.add_argument('--max_iter', type=int, default=2300)
    parser.add_argument('--loss_print_every_n_iter', type=int, default=100)
    parser.add_argument('--gpu_num_per_node', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='node/machine number for training')
    parser.add_argument('--node_ips', type=str_list, default=['192.168.1.13', '192.168.1.14'],
                        help='nodes ip list for training, devided by ",", length >= num_nodes')
    parser.add_argument("--ctrl_port", type=int, default=50051, help='ctrl_port for multinode job')
    parser.add_argument('--hidden_units_num', type=int, default=7)
    parser.add_argument('--hidden_size', type=int, default=1024)

    FLAGS = parser.parse_args()
    
    def _print_args(args):
        import datetime
        print("=".ljust(66, "="))
        print(
            "Running {}: gpu_num_per_node = {}, num_nodes = {}.".format(
                args.model, args.gpu_num_per_node, args.num_nodes
            )
        )
        print("=".ljust(66, "="))
        for arg in vars(args):
            print("{} = {}".format(arg, getattr(args, arg)))
        print("-".ljust(66, "-"))
        print("Time stamp: {}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))
    
    if print_args:
        _print_args(FLAGS)
    return FLAGS

if __name__ == "__main__":
    args = get_args()
    model = WideAndDeep(args)
    model.compile()
    model.summary()
    model.fit(max_iter = args.max_iter, 
              display = args.loss_print_every_n_iter, 
              eval_interval = args.eval_interval, 
              snapshot = 1000000, 
              snapshot_prefix = "wdl")
