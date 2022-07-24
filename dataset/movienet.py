# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import random

import einops
import ndjson
import numpy as np
import torch
from dataset.base import BaseDataset


class MovieNetDataset(BaseDataset):
    def __init__(self, cfg, mode, is_train):
        super(MovieNetDataset, self).__init__(cfg, mode, is_train)

        if mode == "finetune" and not self.use_raw_shot:
            self.shot_repr_dir = os.path.join(
                self.cfg.FEAT_PATH, "swin"
            )

    def load_data(self):
        self.tmpl = "{}/shot_{}_img_{}.jpg"  # video_id, shot_id, shot_num
        if self.mode == "extract_shot":
            with open(
                os.path.join(self.cfg.ANNO_PATH, "anno.trainvaltest.ndjson"), "r"
            ) as f:
                self.anno_data = ndjson.load(f)

        elif self.mode == "pretrain":
            if self.is_train:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.pretrain.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)
            else:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.test.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)

        elif self.mode == "finetune":
            if self.is_train:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.train.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)

                self.vidsid2label = {
                    f"{it['video_id']}_{it['shot_id']}": it["boundary_label"]
                    for it in self.anno_data
                }
            else:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.test.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)

            self.use_raw_shot = self.cfg.USE_RAW_SHOT
            if not self.use_raw_shot:
                self.tmpl = "{}/shot_{}.npy"  # video_id, shot_id

    def _getitem_for_pretrain(self, idx: int):
        data1 = self.anno_data[idx]
        data2 = self.anno_data[idx+1]
        
        idx2 = idx + 1
        if (data1["video_id"] != data2["video_id"]) or (idx2 >= len(self.anno_data)):
            data2 = self.anno_data[idx-1]
            idx2 = idx - 1
        
        payload = {"idx": [idx, idx2],
                   "vid": [data1["video_id"], data2["video_id"]],
                   "sid": [data1["shot_id"], data2["shot_id"]],
                   "num_shot": [data1["num_shot"], data2["num_shot"]]}

        shot_idx1 = self.shot_sampler(int(data1["shot_id"]), data1["num_shot"])
        shot_idx2 = self.shot_sampler(int(data1["shot_id"]), data2["num_shot"])

        video1 = self.load_shot_list(data1["video_id"], shot_idx1)
        video1 = self.apply_transform(video1)
        video2 = self.load_shot_list(data2["video_id"], shot_idx2)
        video2 = self.apply_transform(video2)
        payload["video"] = [video1, video2]

        return payload

    def _getitem_for_extract_shot(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        payload = {"vid": vid, "sid": sid}

        video, s = self.load_shot(vid, sid)
        video = self.apply_transform(video)
        video = einops.rearrange(video, "(s k) c ... -> s k c ...", s=s)
        payload["video"] = video  # [s=1 k c h w]

        assert "video" in payload
        return payload

    def _getitem_for_finetune(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid, sid = data["video_id"], data["shot_id"]
        num_shot = data["num_shot"]

        shot_idx = self.shot_sampler(int(sid), num_shot)

        if self.use_raw_shot:
            video, s = self.load_shot_list(vid, shot_idx)
            video = self.apply_transform(video)
            video = video.view(
                len(shot_idx), 1, -1, 224, 224
            )  # the shape is [S,1,C,H,W]

        else:
            _video = []
            for sidx in shot_idx:
                shot_feat_path = os.path.join(
                    self.shot_repr_dir, self.tmpl.format(vid, f"{sidx:04d}")
                )
                shot = np.load(shot_feat_path)
                shot = torch.from_numpy(shot)
                if len(shot.shape) > 1:
                    shot = shot.mean(0)

                _video.append(shot)
            video = torch.stack(_video, dim=0)

        payload = {
            "idx": idx,
            "vid": vid,
            "sid": sid,
            "video": video,
            "label": abs(data["boundary_label"]),  # ignore -1 label.
        }

        return payload

    def __getitem__(self, idx: int):
        if self.mode == "extract_shot":
            return self._getitem_for_extract_shot(idx)

        elif self.mode == "pretrain":
            return self._getitem_for_pretrain(idx)

        elif self.mode == "finetune":
            return self._getitem_for_finetune(idx)
