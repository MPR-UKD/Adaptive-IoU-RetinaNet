from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union
import torch


class Dataset(ABC):
    @abstractmethod
    def __init__(
        self,
    ):
        self.pos: float
        self.neg: float
        self.adaptive_epochs: int
        self.epoch: Dict
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def num_classes(self) -> List:
        assert AssertionError(
            "This is a necessary function for the functionality of the Code"
        )

    @abstractmethod
    def __len__(self) -> int:
        assert AssertionError(
            "This is a necessary function for the functionality of the Code"
        )

    def adaptive_iou(self):
        # Compute the positive and negative sample ratios based on the current epoch and adaptive epochs
        pos_ratio = (
            self.pos / self.adaptive_epochs * self.epoch["epoch"] + 0.001
            if self.adaptive_epochs != 0
            else self.pos
        )
        neg_ratio = (
            self.neg / self.adaptive_epochs * self.epoch["epoch"]
            if self.adaptive_epochs != 0
            else self.neg
        )

        # Limit the positive and negative sample ratios to the maximum values
        pos_ratio = pos_ratio if pos_ratio < self.pos else self.pos
        neg_ratio = neg_ratio if neg_ratio < self.neg else self.neg

        # Set a minimum positive sample ratio to avoid division by zero
        pos_ratio = pos_ratio if pos_ratio > 0.05 else 0.05

        # Ensure that the positive sample ratio is greater than the negative sample ratio
        assert pos_ratio > neg_ratio

        return pos_ratio, neg_ratio

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        pass
