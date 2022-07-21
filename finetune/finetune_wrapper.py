import os
import json
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from finetune.utils.metric import (
    AccuracyMetric,
    F1ScoreMetric,
    MovieNetMetric,
    SklearnAPMetric,
    SklearnAUCROCMetric,
)
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay


class FinetuningWrapper(pl.LightningModule):
    def __init__(self, cfg, shot_encoder, crn):
        super().__init__()
        
        # define model components
        self.shot_encoder = shot_encoder
        self.crn = crn
        self.head = nn.Sequential(
            nn.Linear(4*768, 2),
            nn.GELU(),
            nn.Dropout(p=0.5),
        )
        if not cfg.MODEL.shot_encoder.enabled:
            self.shot_encoder = None

        # define metrics
        self.acc_metric = AccuracyMetric()
        self.train_ap_metric = SklearnAPMetric()
        self.val_ap_metric = SklearnAPMetric()
        self.f1_metric = F1ScoreMetric()
        self.auc_metric = SklearnAUCROCMetric()
        self.movienet_metric = MovieNetMetric()

        self.cfg = cfg
        self.log_dir = os.path.join(cfg.LOG_PATH, cfg.EXPR_NAME)
        self.eps = 1e-5

    def forward(self, x):
        if self.shot_encoder is not None:
            x = self.shot_encoder(x)
        x = self.crn(x)
        x = einops.rearrange(x, "b w d -> b (w d)")
        x = self.head(x)

        return x

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

    def training_step(self, batch, batch_idx):
        inputs = batch["video"]
        labels = batch["label"]
        outputs = self(inputs)

        loss = F.cross_entropy(outputs.squeeze(), labels.squeeze(), reduction='none')
        lpos = labels == 1
        lneg = labels == 0

        wp = (0.5 * lpos) / (lpos.sum() + self.eps)
        wn = (0.5 * lneg) / (lneg.sum() + self.eps)
        w = wp + wn
        loss = (w * loss).sum()

        self.log(
            "sbd_train/loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        prob = F.softmax(outputs, dim=1)
        self.train_ap_metric.update(prob[:, 1], labels)

        return loss

    def training_epoch_end(self, outputs):
        ap, _, _ = self.train_ap_metric.compute()
        ap *= 100.
        torch.cuda.synchronize()
        self.log(
            "sbd_train/AP",
            ap,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_ap_metric.reset()

    def validation_step(self, batch, batch_idx):
        vids = batch["vid"]
        sids = batch["sid"]
        inputs = batch["video"]
        labels = batch["label"]
        outputs = self(inputs)

        prob = F.softmax(outputs, dim=1)
        preds = torch.argmax(prob, dim=1)

        self.acc_metric.update(prob[:, 1], labels)
        self.val_ap_metric.update(prob[:, 1], labels)
        self.f1_metric.update(prob[:, 1], labels)
        self.auc_metric.update(prob[:, 1], labels)
        for vid, sid, pred, gt in zip(vids, sids, preds, labels):
            self.movienet_metric.update(vid, sid, pred, gt)

    def validation_epoch_end(self, outputs):
        score = {}

        # update acc.
        acc = self.acc_metric.compute()
        torch.cuda.synchronize()
        assert isinstance(acc, dict)
        score.update(acc)

        # update average precision (AP).
        ap, _, _ = self.val_ap_metric.compute() 
        ap *= 100.0
        torch.cuda.synchronize()
        assert isinstance(ap, torch.Tensor)
        score.update({"ap": ap})

        # update AUC-ROC
        auc, _, _ = self.auc_metric.compute()
        auc *= 100.0
        torch.cuda.synchronize()
        assert isinstance(auc, torch.Tensor)
        score.update({"auc": auc})

        # update F1 score.
        f1 = self.f1_metric.compute() * 100.0
        torch.cuda.synchronize()
        assert isinstance(f1, torch.Tensor)
        score.update({"f1": f1})

        # update recall, mIoU score.
        recall, recall_at_3s, miou = self.movienet_metric.compute()
        torch.cuda.synchronize()
        assert isinstance(recall, torch.Tensor)
        assert isinstance(recall_at_3s, torch.Tensor)
        assert isinstance(miou, torch.Tensor)
        score.update({"recall": recall * 100.0})
        score.update({"recall@3s": recall_at_3s * 100})
        score.update({"mIoU": miou * 100})

        # logging
        for k, v in score.items():
            self.log(
                f"sbd_test/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        score = {k: v.item() for k, v in score.items()}
        self.print(f"\nTest Score: {score}")

        # reset all metrics
        self.acc_metric.reset()
        self.val_ap_metric.reset()
        self.f1_metric.reset()
        self.auc_metric.reset()
        self.movienet_metric.reset()

        # save last epoch result.
        with open(os.path.join(
            self.log_dir, 
            f"all_score_epoch_{self.current_epoch}.json"),
            "w") as fopen:
            json.dump(score, fopen, indent=4, ensure_ascii=False)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)