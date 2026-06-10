# Copyright (c) Megvii Inc. All rights reserved.
import os
import torch  # 🚨 新增：用于加载权重
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_summary import ModelSummary

from callbacks.ema import EMACallback
from utils.torch_dist import all_gather_object, synchronize

from .base_exp import BEVDepthLightningModel


def run_cli(model_class=BEVDepthLightningModel,
            exp_name='base_exp',
            use_ema=False,
            ckpt_path=None):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-p',
                               '--predict',
                               dest='predict',
                               action='store_true',
                               help='predict model on testing set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=24,# 24
                        # strategy='ddp',
                        # strategy='single_device',
                        # strategy='ddp_find_unused_parameters_false',
                        num_sanity_val_steps=0,
                        check_val_every_n_epoch=1,
                        gradient_clip_val=5,
                        limit_val_batches=0.25,
                        log_every_n_steps=50,
                        enable_checkpointing=True,
                        precision=16,
                        default_root_dir=os.path.join('./outputs/', exp_name))
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))
    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback, ModelSummary(max_depth=3)],
        accelerator="gpu",
        devices=[0]
        )
    else:
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ModelSummary(max_depth=3)],
        accelerator="gpu",
        devices=[0])
        
    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    elif args.predict:
        predict_step_outputs = trainer.predict(model, ckpt_path=args.ckpt_path)
        all_pred_results = list()
        all_img_metas = list()
        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])
        synchronize()
        len_dataset = len(model.test_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        model.evaluator._format_bbox(all_pred_results, all_img_metas,
                                     os.path.dirname(args.ckpt_path))
    # else:
    #     # =====================================================================
    #     #  增量训练 / 微调 
    #     # =====================================================================
    #     # 之前训练好的最优权重路径
    #     pretrained_ckpt = "outputs/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_81/checkpoints/epoch=23-step=4320.ckpt"
        
    #     if os.path.exists(pretrained_ckpt):
    #         print(f" [增量训练] 正在提取记忆 (权重): {pretrained_ckpt}")
    #         checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
            
    #         # 严格把权重灌入当前模型中
    #         model.load_state_dict(checkpoint['state_dict'], strict=True)
    #         print(" 权重灌注成功！即将以全新的优化器状态，从 Epoch 0 开始增量微调...")
    #     else:
    #         print(f"⚠️ 警告：找不到权重文件 {pretrained_ckpt}，将从零开始训练！")

    #     # 注意：这里千万不要加 ckpt_path 参数，让 Lightning 以为这是一个全新的训练任务
    #     trainer.fit(model)
    #     # =====================================================================
    else:
        print("未加载任何历史检测权重")
        trainer.fit(model, ckpt_path=args.ckpt_path)