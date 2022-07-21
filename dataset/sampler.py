# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import random
import numpy as np

class NeighborShotSampler:
    """ This is for scene boundary detection (sbd), i.e., fine-tuning stage """
    def __init__(self, neighbor_size: int = 8):
        self.neighbor_size = neighbor_size

    def __call__(self, center_sid: int, total_num_shot: int):
        # total number of shots = 2 * neighbor_size
        shot_idx = center_sid + np.arange(-self.neighbor_size + 1, self.neighbor_size + 1)
        shot_idx = np.clip(shot_idx, 0, total_num_shot)

        return shot_idx
