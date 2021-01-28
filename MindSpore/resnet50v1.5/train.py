# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train resnet."""
import os
import argparse
import ast
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from src.lr_generator import get_lr, warmup_cosine_annealing_lr
from src.CrossEntropySmooth import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default=None, help='Resnet Model, either resnet50 or resnet101')
parser.add_argument('--dataset', type=str, default=None, help='Dataset, either cifar10 or imagenet2012')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')

parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
parser.add_argument("--train_steps", type=int, default=-1, help="Training Steps, default is -1, "
                                                                "meaning run all steps according to epoch number.")
parser.add_argument("--data_sink_steps", type=int, default="1", help="Sink steps for each epoch, default is 1.")
parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="dtype, default is fp32.")

args_opt = parser.parse_args()
from mindspore import log as logger
logger.warning("\nargs_opt: {}".format(args_opt))

set_seed(1)

if args_opt.net == "resnet50":
    from src.resnet import resnet50 as resnet
    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.config import config2 as config
        from src.dataset import create_dataset2 as create_dataset
elif args_opt.net == "resnet101":
    from src.resnet import resnet101 as resnet
    from src.config import config3 as config
    from src.dataset import create_dataset3 as create_dataset
else:
    from src.resnet import se_resnet50 as resnet
    from src.config import config4 as config
    from src.dataset import create_dataset4 as create_dataset


if __name__ == '__main__':
    target = args_opt.device_target
    if target == "CPU":
        args_opt.run_distribute = False

    config.save_checkpoint = False
    config.epoch_size = 1
    config.batch_size = args_opt.batch_size
    config.momentum = 0.875
    config.lr_init = 0.001
    amp_level = "O2" if args_opt.dtype == "fp16" else "O0"
    ckpt_save_dir = config.save_checkpoint_path

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args_opt.parameter_server:
        context.set_ps_context(enable_ps=True)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            if args_opt.net == "resnet50" or args_opt.net == "se-resnet50":
                context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
            else:
                context.set_auto_parallel_context(all_reduce_fusion_config=[180, 313])
            init()
        # GPU target
        else:
            init()
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            if args_opt.net == "resnet50":
                context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(get_rank()) + "/"

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target, distribute=args_opt.run_distribute)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet(class_num=config.class_num)
    if args_opt.parameter_server:
        net.set_param_ps()

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    # init lr
    if args_opt.net == "resnet50" or args_opt.net == "se-resnet50":
        lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                    warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                    lr_decay_mode=config.lr_decay_mode)
    else:
        lr = warmup_cosine_annealing_lr(config.lr, step_size, config.warmup_epochs, config.epoch_size,
                                        config.pretrain_epoch_size * step_size)
    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    # define loss, model
    if target == "Ascend":
        if args_opt.dataset == "imagenet2012":
            if not config.use_label_smooth:
                config.label_smooth_factor = 0.0
            loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                      smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
        else:
            loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=False)
    else:
        # GPU and CPU target
        if args_opt.dataset == "imagenet2012":
            if not config.use_label_smooth:
                config.label_smooth_factor = 0.0
            loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                      smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
        else:
            loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

        if (args_opt.net == "resnet101" or args_opt.net == "resnet50") and \
            not args_opt.parameter_server and target != "CPU":
            opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum, config.weight_decay,
                           config.loss_scale)
            from mindspore.train.loss_scale_manager import DynamicLossScaleManager
            loss_scale = DynamicLossScaleManager()
            model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                          amp_level=amp_level, keep_batchnorm_fp32=False)
        else:
            ## fp32 training
            opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum, config.weight_decay)
            model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    # define callbacks
    time_cb = TimeMonitor(data_size=args_opt.data_sink_steps)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    if args_opt.net == "se-resnet50":
        config.epoch_size = config.train_epoch_size
    dataset_sink_mode = (not args_opt.parameter_server) and target != "CPU"

    assert(args_opt.train_steps <= dataset.get_dataset_size())
    train_steps = dataset.get_dataset_size() if args_opt.train_steps < 0 else args_opt.train_steps
    new_repeat_count = (config.epoch_size - config.pretrain_epoch_size) * train_steps // args_opt.data_sink_steps
    model.train(new_repeat_count, dataset, callbacks=cb,
                sink_size=args_opt.data_sink_steps, dataset_sink_mode=dataset_sink_mode)
