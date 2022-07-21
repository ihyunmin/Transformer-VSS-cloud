import os
import yaml
import random
import numpy as np
import torch
import pytorch_lightning as pl
from finetune.utils.attr_dict import AttrDict
from dataset import (
    get_dataset,
    get_collate_fn,
)
from model import (
    get_shot_encoder,
    get_contextual_relation_network,
)
from finetune.finetune_wrapper import FinetuningWrapper


def init_config(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)
        cfg = AttrDict(cfg)

    # set random seed
    pl.seed_everything(cfg.SEED)

    cfg.PROJ_ROOT = "/data/private/train"

    # dataset path
    if cfg.DATASET == "movienet":
        cfg.DATA_PATH = "/data/project/rw/movienet"
        cfg.IMG_PATH = os.path.join(cfg.DATA_PATH, "240P_frames")
        cfg.FEAT_PATH = os.path.join(cfg.DATA_PATH, "features")
        cfg.ANNO_PATH = os.path.join(cfg.DATA_PATH, "anno")
    else:
        raise NotImplementedError

    # checkpoint path
    cfg.CKPT_PATH = os.path.join(
        cfg.PROJ_ROOT, "finetune/ckpt"
    )
    cfg.PRETRAINED_CKPT_PATH = os.path.join(
        cfg.PROJ_ROOT, "pretrain/ckpt"
    )

    # log path
    cfg.LOG_PATH = os.path.join(
        cfg.PROJ_ROOT, "finetune/logs"
    )

    # distributed
    cfg.DISTRIBUTED.WORLD_SIZE = int(
        cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    )
    assert cfg.TRAIN.BATCH_SIZE % cfg.DISTRIBUTED.WORLD_SIZE == 0
    cfg.TRAIN.BATCH_SIZE = int(
        cfg.TRAIN.BATCH_SIZE / cfg.DISTRIBUTED.WORLD_SIZE
    )
    cfg.TRAINER.gpus = cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    cfg.TRAINER.num_nodes = cfg.DISTRIBUTED.NUM_NODES
    if cfg.MODEL.use_sync_bn:
        cfg.TRAINER.sync_batchnorm = True

    # auto scaling learning rate
    cfg.TRAIN.OPTIMIZER.lr.scaled_lr = cfg.TRAIN.OPTIMIZER.lr.base_lr
    if cfg.TRAIN.OPTIMIZER.lr.auto_scale:
        cfg.TRAIN.OPTIMIZER.lr.scaled_lr = (
            cfg.TRAIN.OPTIMIZER.lr.base_lr
            * cfg.TRAIN.BATCH_SIZE
            / float(cfg.TRAIN.OPTIMIZER.lr.base_lr_batch_size)
        )

    return cfg


def load_pretrained_config(cfg):
    ...


def init_data_loader(cfg, mode, is_train):
    data_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(cfg, mode=mode, is_train=is_train),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        drop_last=True, # modified to False when val/test
        shuffle=is_train,
        collate_fn=get_collate_fn(cfg),
    )

    if is_train:
        cfg.TRAIN.TRAIN_ITERS_PER_EPOCH = (
            len(data_loader.dataset) // cfg.TRAIN.BATCH_SIZE
        )

    return cfg, data_loader


def init_model(cfg):
    shot_encoder = get_shot_encoder(cfg)
    crn = get_contextual_relation_network(cfg)

    # fine-tuning까지 끝난 것을 더 학습시킬 때
    if "LOAD_FROM" in cfg and len(cfg.LOAD_FROM) > 0:
        print("LOAD SBD MODEL WEIGHTS FROM: ", cfg.LOAD_FROM)
        model = FinetuningWrapper.load_from_checkpoint(
            cfg=cfg,
            shot_encoder=shot_encoder,
            crn=crn,
            # 모델 이름(model-v1.ckpt) 수정 필요
            checkpoint_path=os.path.join(cfg.CKPT_PATH, cfg.LOAD_FROM, "model-v1.ckpt"),
            strict=False,
        )
    # pre-training을 끝내고 fine-tuning을 하기 위해
    elif "PRETRAINED_LOAD_FROM" in cfg and len(cfg.PRETRAINED_LOAD_FROM) > 0:
        print("LOAD PRETRAINED MODEL WEIGHTS FROM: ", cfg.PRETRAINED_LOAD_FROM)
        model = FinetuningWrapper.load_from_checkpoint(
            cfg=cfg,
            shot_encoder=shot_encoder,
            crn=crn,
            checkpoint_path=os.path.join(
                # 모델 이름(model-v1.ckpt) 수정 필요
                cfg.PRETRAINED_CKPT_PATH, cfg.PRETRAINED_LOAD_FROM, "model-v1.ckpt"
            ),
            strict=False,
        )
    else:
        model = FinetuningWrapper(cfg, shot_encoder, crn)

    return cfg, model


def init_trainer(cfg):
    logger = None
    callbacks = []

    # logging
    log_path = os.path.join(
        cfg.LOG_PATH, cfg.EXPR_NAME
    )
    os.makedirs(log_path, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(log_path)

    # checkpoint callback
    ckpt_path = os.path.join(
        cfg.CKPT_PATH, cfg.EXPR_NAME
    )
    os.makedirs(ckpt_path, exist_ok=True)
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, 
            monitor=None, 
            save_top_k=-1,
            filename="model_{epoch:02d}",
            every_n_val_epochs=1,
        )
    )

    # learning rate callback
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    # GPU stat callback
    callbacks.append(
        pl.callbacks.GPUStatsMonitor(
            memory_utilization=True,
            gpu_utilization=True,
            intra_step_time=True,
            inter_step_time=True,
        )
    )

    trainer = pl.Trainer(**cfg.TRAINER, callbacks=callbacks, logger=logger)

    return cfg, trainer