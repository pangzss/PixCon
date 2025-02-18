# Copyright (c) OpenMMLab. All rights reserved.
from math import cos, pi

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module(name=['PowerUpdateHook'])
class PowerUpdateHook(Hook):
    def __init__(self, end_power=0.5, update_interval=1, **kwargs):
        self.end_power = end_power
        self.update_interval = update_interval

    def before_train_epoch(self, runner):
        assert hasattr(runner.model.module, 'power'), \
            "The runner must have attribute \"power\" in algorithms."
        assert hasattr(runner.model.module, 'base_power'), \
            "The runner must have attribute \"base_power\" in algorithms."
        if self.every_n_epochs(runner, self.update_interval):
            power_delta = (self.end_power - runner.model.module.base_power) / runner.max_epochs
            runner.model.module.power = runner.model.module.power + power_delta
