import os
import time
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import matplotlib as plt

from mmcv.runner import RUNNERS, EpochBasedRunner
from mmcv.runner import get_dist_info
from mmcv import ProgressBar
from mmcv.runner import HOOKS, Hook, TensorboardLoggerHook
from mmcv.runner.dist_utils import master_only

@RUNNERS.register_module()
class EpochBasedRunnerProgressBar(EpochBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader

        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = ProgressBar(len(data_loader))
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(self.data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1
            if rank == 0:
                prog_bar.update()
                
        self.call_hook('after_train_epoch')
        self._epoch += 1