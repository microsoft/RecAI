# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from torch import Tensor
from transformers import BatchEncoding


@dataclass
class RewardOutput:
    reward: list[float]
    complete_data: BatchEncoding = None
    action_mask: Tensor = None
    total_reward: Tensor = None


class BaseRewardModel:
    def get_reward(self, batch, model_output, only_reward=False) -> RewardOutput:
        ...
