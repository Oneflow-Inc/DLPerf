import hugectr
from mpi4py import MPI

def DLRM(args):
    vvgpu = [[g for g in range(args.gpu_num_per_node)] for n in range(args.num_nodes)]
    solver = hugectr.CreateSolver(max_eval_batches = args.eval_batchs,
                                batchsize_eval = args.batch_size,
                                batchsize = args.batch_size,
                                lr = args.learning_rate,
                                warmup_steps = args.warmup_steps,
                                decay_start = args.decay_start,
                                decay_steps = args.decay_steps,
                                decay_power = args.decay_power,
                                end_lr = args.end_lr,
                                vvgpu = vvgpu,
                                repeat_dataset = True)
    reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Raw,
                                    source = [f"{args.data_dir}/train_data.bin"],
                                    eval_source = f"{args.data_dir}/test_data.bin",
                                    num_samples = 36672493,
                                    eval_num_samples = 4584062,
                                    check_type = hugectr.Check_t.Non)
    optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,
                                        update_type = hugectr.Update_t.Local,
                                        atomic_update = True)
    model = hugectr.Model(solver, reader, optimizer)
    model.add(hugectr.Input(label_dim = 1, label_name = "label",
                            dense_dim = 13, dense_name = "dense",
                            data_reader_sparse_param_array = 
                            [hugectr.DataReaderSparseParam("data1", 2, False, 26)]))
    model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.LocalizedSlotSparseEmbeddingOneHot, 
                                slot_size_array = [1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572],
                                workspace_size_per_gpu_in_mb = args.workspace_size_per_gpu_in_mb,
                                embedding_vec_size = args.embedding_vec_size,
                                combiner = "sum",
                                sparse_embedding_name = "sparse_embedding1",
                                bottom_name = "data1",
                                optimizer = optimizer))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["dense"],
                                top_names = ["fc1"],
                                num_output=512))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc1"],
                                top_names = ["relu1"]))                           
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu1"],
                                top_names = ["fc2"],
                                num_output=256))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc2"],
                                top_names = ["relu2"]))                            
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu2"],
                                top_names = ["fc3"],
                                num_output=128))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc3"],
                                top_names = ["relu3"]))                              
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,
                                bottom_names = ["relu3","sparse_embedding1"],
                                top_names = ["interaction1"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["interaction1"],
                                top_names = ["fc4"],
                                num_output=1024))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc4"],
                                top_names = ["relu4"]))                              
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu4"],
                                top_names = ["fc5"],
                                num_output=1024))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc5"],
                                top_names = ["relu5"]))                              
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu5"],
                                top_names = ["fc6"],
                                num_output=512))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc6"],
                                top_names = ["relu6"]))                               
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu6"],
                                top_names = ["fc7"],
                                num_output=256))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc7"],
                                top_names = ["relu7"]))                                                                              
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu7"],
                                top_names = ["fc8"],
                                num_output=1))                                                                                           
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                                bottom_names = ["fc8", "label"],
                                top_names = ["loss"]))
    return model

def get_args(print_args=True):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num_per_node', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='node/machine number for training')
    parser.add_argument('--eval_batchs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=65536)
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--warmup_steps', type=int, default=300)
    parser.add_argument('--decay_start', type=int, default=0)
    parser.add_argument('--decay_steps', type=int, default=1)
    parser.add_argument('--decay_power', type=int, default=2)
    parser.add_argument('--end_lr', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='/dataset/f9f659c5/hugectr_dlrm')
    parser.add_argument('--workspace_size_per_gpu_in_mb', type=int, default=11645)
    parser.add_argument('--embedding_vec_size', type=int, default=128)
    parser.add_argument('--max_iter', type=int, default=600)
    parser.add_argument('--loss_print_every_n_iter', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1000)


    FLAGS = parser.parse_args()

    def _print_args(args):
        from datetime import datetime
        print("=".ljust(66, "="))
        print(
            "Running {}: gpu_num_per_node = {}, num_nodes = {}.".format(
                "HugeCTR-WDL", args.gpu_num_per_node, args.num_nodes
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
    model=DLRM(args)
    model.compile()
    model.summary()
    model.fit(
        max_iter = args.max_iter, 
        display = args.loss_print_every_n_iter, 
        eval_interval = args.eval_interval, 
        snapshot = 10000000, 
        snapshot_prefix = "dlrm")
