# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from .crn.swin import SwinTransformerCRN
from .crn.pool import TemporalPooling
from .shot_encoder.resnet import resnet50


def get_shot_encoder(cfg):
    name = cfg.MODEL.shot_encoder.name
    shot_encoder_args = cfg.MODEL.shot_encoder[name]
    if name == "resnet":
        depth = shot_encoder_args["depth"]
        if depth == 50:
            shot_encoder = resnet50(
                pretrained=shot_encoder_args["use_imagenet_pretrained"],
                **shot_encoder_args["params"],
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return shot_encoder


def get_contextual_relation_network(cfg):
    name = cfg.MODEL.contextual_relation_network.name
    crn_args = cfg.MODEL.contextual_relation_network[name]
    
    if name == 'swin':
        crn = SwinTransformerCRN(
                nn_size=cfg.MODEL.neighbor_size * 2,
                num_layers=crn_args['num_layers'],
                input_dim=crn_args['input_dim'],
                d_model=crn_args['d_model'],
                num_heads=crn_args['num_heads'],
                batch_size=cfg.TRAIN.BATCH_SIZE,
                window_size=crn_args['window_size'],
                hidden_dropout_prob=crn_args['hidden_dropout_prob'])
    elif name == 'pool':
        crn = TemporalPooling(
                d_model=crn_args['d_model'],
                dropout_ratio=crn_args['dropout_ratio'])
    else:
        raise NotImplementedError
        
    return crn


__all__ = ["get_shot_encoder", "get_contextual_relation_network"]
