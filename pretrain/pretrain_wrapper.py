import os
import einops
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pretrain.loss.nce_loss import (
    #compute_intra_window_nce_loss,
    #compute_inter_shot_nce_loss,
    compute_info_nce_loss,
)


class PretrainingWrapper(pl.LightningModule):
    def __init__(self, cfg, shot_encoder, crn):
        super().__init__()

        self.cfg = cfg
        self.shot_encoder = shot_encoder
        self.crn = crn

        self.K = cfg.MODEL.neighbor_size
        self.T = cfg.TRAIN.TEMPERATURE

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        # params
        skip_list = []
        weight_decay = self.cfg.TRAIN.OPTIMIZER.weight_decay
        if not self.cfg.TRAIN.OPTIMIZER.regularize_bn:
            skip_list.append("bn")
        if not self.cfg.TRAIN.OPTIMIZER.regularize_bias:
            skip_list.append("bias")
        params = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=weight_decay, skip_list=skip_list
        )

        # optimizer
        if self.cfg.TRAIN.OPTIMIZER.name == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif self.cfg.TRAIN.OPTIMIZER.name == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr
            )
        elif self.cfg.TRAIN.OPTIMIZER.name == "lars":
            optimizer = LARS(
                params,
                lr=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr,
                momentum=0.9,
                weight_decay=weight_decay,
                trust_coefficient=0.001,
            )
        else:
            raise ValueError()

        # learning rate scheduler
        warmup_steps = int(
            self.cfg.TRAIN.TRAIN_ITERS_PER_EPOCH
            * self.cfg.TRAINER.max_epochs
            * self.cfg.TRAIN.OPTIMIZER.scheduler.warmup
        )
        total_steps = int(
            self.cfg.TRAIN.TRAIN_ITERS_PER_EPOCH * self.cfg.TRAINER.max_epochs
        )

        if self.cfg.TRAIN.OPTIMIZER.scheduler.name == "cosine_with_linear_warmup":
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.cfg.TRAIN.OPTIMIZER.scheduler.name == "cosine_annealing":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=self.cfg.TRAINER.max_epochs,
                                                       eta_min=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr/50)
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]

    def extract_shot_representation(self, inputs, is_train=True):
        if is_train:
            b, s, c, h, w = inputs.shape
            inputs = einops.rearrange(inputs, "b s c h w -> (b s) c h w")
            shot_repr = self.shot_encoder(inputs)
            shot_repr = einops.rearrange(shot_repr, "(b s) d -> b s d", s=s)
        else:
            b, s, k, c, h, w = inputs.shape
            inputs = einops.rearrange(inputs, "b s k c h w -> (b s) k c h w", s=s)
            keyframe_repr = [self.shot_encoder(inputs[:, _k]) for _k in range(k)]
            shot_repr = torch.stack(keyframe_repr).mean(dim=0)  # [k (b s) d] -> [(b s) d]

        return shot_repr

    def training_step(self, batch, batch_idx):
        inputs = batch["video"]
        inputs = torch.cat(inputs, dim=0)

        shot_repr = self.extract_shot_representation(inputs, is_train=True)
        feats = self.crn(shot_repr)

        loss = compute_info_nce_loss(feats, self.T)
    
        self.log(
            "pretrain/loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        '''
        extract shot representation per every epoch
        '''

        vids = batch["vid"]
        sids = batch["sid"]

        assert len(batch["video"].shape) == 6 # b s k c h w
        inputs = batch["video"]

        x = self.extract_shot_representation(inputs, is_train=False)
        embedding = x.float().cpu().numpy()

        for vid, sid, feat in zip(vids, sids, embedding):
            os.makedirs(
                os.path.join(
                    self.cfg.SHOT_LOG_PATH, 
                    self.cfg.LOAD_FROM,
                    str(self.current_epoch),
                    vid), exist_ok=True
            )
            new_filename = f"{vid}/shot_{sid}"
            new_filepath = os.path.join(
                self.cfg.SHOT_LOG_PATH, 
                self.cfg.LOAD_FROM,
                str(self.current_epoch),
                new_filename,
            )
            np.save(new_filepath, feat)

    def test_step(self, batch, batch_idx):
        '''
        extract shot representation for finetuning
        do not update params in shot encoder during finetuning
        '''

        vids = batch["vid"]
        sids = batch["sid"]

        assert len(batch["video"].shape) == 6 # b s k c h w
        inputs = batch["video"]

        x = self.extract_shot_representation(inputs, is_train=False)
        embedding = x.float().cpu().numpy()

        for vid, sid, feat in zip(vids, sids, embedding):
            os.makedirs(
                os.path.join(self.cfg.FEAT_PATH, self.cfg.LOAD_FROM, vid), exist_ok=True
            )
            new_filename = f"{vid}/shot_{sid}"
            new_filepath = os.path.join(
                self.cfg.FEAT_PATH, self.cfg.LOAD_FROM, new_filename
            )
            np.save(new_filepath, feat)