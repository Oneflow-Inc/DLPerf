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
"""
#################pre_train bert example on zh-wiki########################
python run_pretrain.py
"""

import os
import argparse
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.optim import Lamb, Momentum, AdamWeightDecay
from mindspore import log as logger
from mindspore.common import set_seed
from src import BertNetworkWithLoss, BertTrainOneStepCell, BertTrainOneStepWithLossScaleCell, \
                BertTrainAccumulateStepsWithLossScaleCell, BertTrainOneStepWithLossScaleCellForAdam, \
                AdamWeightDecayForBert
from src.dataset import create_bert_dataset
from src.config import cfg, bert_net_cfg
from src.utils import LossCallBack, BertLearningRate
_current_dir = os.path.dirname(os.path.realpath(__file__))


def _set_bert_all_reduce_split(device_target='Ascend', enable_graph_kernel=False):
    """set bert all_reduce fusion split, support num_hidden_layers is 12 and 24."""
    if bert_net_cfg.num_hidden_layers == 12:
        if bert_net_cfg.use_relative_positions:
            context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 87, 116, 145, 174, 203, 217])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[28, 55, 82, 109, 136, 163, 190, 205])
            if device_target == 'GPU' and enable_graph_kernel:
                context.set_auto_parallel_context(all_reduce_fusion_config=[180, 205])
    elif bert_net_cfg.num_hidden_layers == 24:
        if bert_net_cfg.use_relative_positions:
            context.set_auto_parallel_context(all_reduce_fusion_config=[30, 90, 150, 210, 270, 330, 390, 421])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[38, 93, 148, 203, 258, 313, 368, 397])


def _get_optimizer(args_opt, network):
    """get bert optimizer, support Lamb, Momentum, AdamWeightDecay."""
    if cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=cfg.Lamb.learning_rate,
                                       end_learning_rate=cfg.Lamb.end_learning_rate,
                                       warmup_steps=cfg.Lamb.warmup_steps,
                                       decay_steps=args_opt.train_steps,
                                       power=cfg.Lamb.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.Lamb.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.Lamb.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.Lamb.weight_decay},
                        {'params': other_params},
                        {'order_params': params}]
        optimizer = Lamb(group_params, learning_rate=lr_schedule, eps=cfg.Lamb.eps)
    elif cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=cfg.Momentum.learning_rate,
                             momentum=cfg.Momentum.momentum)
    elif cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=cfg.AdamWeightDecay.warmup_steps,
                                       decay_steps=args_opt.train_steps,
                                       power=cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0},
                        {'order_params': params}]
        if args_opt.enable_lossscale == "true" and args_opt.device_target == 'GPU':
            optimizer = AdamWeightDecayForBert(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
        else:
            optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
    else:
        raise ValueError("Don't support optimizer {}, only support [Lamb, Momentum, AdamWeightDecay]".
                         format(cfg.optimizer))
    return optimizer


def _auto_enable_graph_kernel(device_target, graph_kernel_mode):
    """Judge whether is suitable to enable graph kernel."""
    return graph_kernel_mode in ("auto", "true") and device_target == 'GPU' and \
        cfg.bert_network == 'base' and cfg.optimizer == 'AdamWeightDecay'


def run_pretrain():
    """pre-train bert_clue"""
    parser = argparse.ArgumentParser(description='bert pre_training')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--epoch_size", type=int, default="1", help="Epoch size, default is 1.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--enable_save_ckpt", type=str, default="true", choices=["true", "false"],
                        help="Enable save checkpoint, default is true.")
    parser.add_argument("--enable_lossscale", type=str, default="true", choices=["true", "false"],
                        help="Use lossscale or not, default is not.")
    parser.add_argument("--do_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable shuffle for dataset, default is true.")
    parser.add_argument("--enable_data_sink", type=str, default="true", choices=["true", "false"],
                        help="Enable data sink, default is true.")
    parser.add_argument("--data_sink_steps", type=int, default="1", help="Sink steps for each epoch, default is 1.")
    parser.add_argument("--accumulation_steps", type=int, default="1",
                        help="Accumulating gradients N times before weight update, default is 1.")
    parser.add_argument("--save_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--load_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--save_checkpoint_steps", type=int, default=1000, help="Save checkpoint steps, "
                                                                                "default is 1000.")
    parser.add_argument("--train_steps", type=int, default=-1, help="Training Steps, default is -1, "
                                                                    "meaning run all steps according to epoch number.")
    parser.add_argument("--save_checkpoint_num", type=int, default=1, help="Save checkpoint numbers, default is 1.")
    parser.add_argument("--data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_dir", type=str, default="", help="Schema path, it is better to use absolute path")
    parser.add_argument("--enable_graph_kernel", type=str, default="auto", choices=["auto", "true", "false"],
                        help="Accelerate by graph kernel, default is auto.")

    parser.add_argument("--optimizer", type=str, default="AdamWeightDecay", choices=["AdamWeightDecay", "Lamb", "Momentum"],
                        help="Optimizer, default is AdamWeightDecay.")
    parser.add_argument("--enable_global_norm", type=str, default="true", choices=["true", "false"],
                        help="Enable gloabl norm for grad clip, default is true.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"],
                        help="dtype, default is fp32.")

    args_opt = parser.parse_args()
    cfg.optimizer = args_opt.optimizer
    cfg.batch_size = args_opt.batch_size
    cfg.enable_global_norm = True if args_opt.enable_global_norm == "true" else False
    bert_net_cfg.compute_type = mstype.float32 if args_opt.dtype== "fp32" else mstype.float16
    logger.warning("\nargs_opt: {}".format(args_opt))
    logger.warning("\ncfg: {}".format(cfg))

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    context.set_context(reserve_class_name_in_scope=False)
    is_auto_enable_graph_kernel = _auto_enable_graph_kernel(args_opt.device_target, args_opt.enable_graph_kernel)
    if args_opt.enable_graph_kernel == "true" or is_auto_enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    ckpt_save_dir = args_opt.save_checkpoint_path
    if args_opt.distribute == "true":
        if args_opt.device_target == 'Ascend':
            D.init()
            device_num = args_opt.device_num
            rank = args_opt.device_id % device_num
        else:
            D.init()
            device_num = D.get_group_size()
            rank = D.get_rank()
        ckpt_save_dir = args_opt.save_checkpoint_path + 'ckpt_' + str(get_rank()) + '/'

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        _set_bert_all_reduce_split(args_opt.device_target, context.get_context('enable_graph_kernel'))
    else:
        rank = 0
        device_num = 1

    if args_opt.accumulation_steps > 1:
        logger.info("accumulation steps: {}".format(args_opt.accumulation_steps))
        logger.info("global batch size: {}".format(cfg.batch_size * args_opt.accumulation_steps))
        if args_opt.enable_data_sink == "true":
            args_opt.data_sink_steps *= args_opt.accumulation_steps
            logger.info("data sink steps: {}".format(args_opt.data_sink_steps))
        if args_opt.enable_save_ckpt == "true":
            args_opt.save_checkpoint_steps *= args_opt.accumulation_steps
            logger.info("save checkpoint steps: {}".format(args_opt.save_checkpoint_steps))

    ds = create_bert_dataset(device_num, rank, args_opt.do_shuffle, args_opt.data_dir, args_opt.schema_dir)
    net_with_loss = BertNetworkWithLoss(bert_net_cfg, True)

    new_repeat_count = args_opt.epoch_size * ds.get_dataset_size() // args_opt.data_sink_steps
    if args_opt.train_steps > 0:
        train_steps = args_opt.train_steps * args_opt.accumulation_steps
        new_repeat_count = min(new_repeat_count, train_steps // args_opt.data_sink_steps)
    else:
        args_opt.train_steps = args_opt.epoch_size * ds.get_dataset_size() // args_opt.accumulation_steps
        logger.info("train steps: {}".format(args_opt.train_steps))

    optimizer = _get_optimizer(args_opt, net_with_loss)
    callback = [TimeMonitor(args_opt.data_sink_steps), LossCallBack(ds.get_dataset_size())]
    if args_opt.enable_save_ckpt == "true" and args_opt.device_id % min(8, device_num) == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoint_steps,
                                     keep_checkpoint_max=args_opt.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_bert',
                                     directory=None if ckpt_save_dir == "" else ckpt_save_dir, config=config_ck)
        callback.append(ckpoint_cb)

    if args_opt.load_checkpoint_path:
        param_dict = load_checkpoint(args_opt.load_checkpoint_path)
        load_param_into_net(net_with_loss, param_dict)

    if args_opt.enable_lossscale == "true":
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                 scale_factor=cfg.scale_factor,
                                                 scale_window=cfg.scale_window)

        if args_opt.accumulation_steps <= 1:
            if cfg.optimizer == 'AdamWeightDecay' and args_opt.device_target == 'GPU':
                net_with_grads = BertTrainOneStepWithLossScaleCellForAdam(net_with_loss, optimizer=optimizer,
                                                                          scale_update_cell=update_cell)
            else:
                net_with_grads = BertTrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                                   scale_update_cell=update_cell)
        else:
            accumulation_steps = args_opt.accumulation_steps
            net_with_grads = BertTrainAccumulateStepsWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                                       scale_update_cell=update_cell,
                                                                       accumulation_steps=accumulation_steps,
                                                                       enable_global_norm=cfg.enable_global_norm)
    else:
        net_with_grads = BertTrainOneStepCell(net_with_loss, optimizer=optimizer)

    model = Model(net_with_grads)
    model.train(new_repeat_count, ds, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == "true"), sink_size=args_opt.data_sink_steps)


if __name__ == '__main__':
    set_seed(0)
    run_pretrain()
