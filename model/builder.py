import argparse

import torch
from .act_vae import build as build_ACT_model
from .act_vae import build_cnnmlp as build_CNNMLP_model


def get_default_model_args():
    return argparse.Namespace(
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        backbone='resnet18',
        dilation=False,
        position_embedding='sine',
        camera_names=[],
        enc_layers=4,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=400,
        pre_norm=False,
        masks=False,
    )


def _build_args(args_override):
    args = get_default_model_args()
    for k, v in args_override.items():
        setattr(args, k, v)
    return args


def build_ACT_model_and_optimizer(args_override):
    args = _build_args(args_override)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    args = _build_args(args_override)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer
