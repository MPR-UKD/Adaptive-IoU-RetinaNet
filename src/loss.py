import torch.nn as nn
import torch.nn.functional as F
from typing import List
import torch
from torchvision.ops import sigmoid_focal_loss

from src.utils import one_hot_embedding


class FocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: List[int],
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initializes the FocalLoss class.

        Args:
            num_classes (int): Number of classes.
            alpha (float): Value of alpha for computing Focal Loss.
            gamma (float): Value of gamma for computing Focal Loss.
            reduction (str): Method for reducing the loss over all data points.
                             Can be 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def focal_loss(self, x, y, n_cls, gamma=None):

        t = one_hot_embedding(y, n_cls)
        if x.is_cuda:
            t = t.cuda()
        return sigmoid_focal_loss(
            x.float(),
            t.float(),
            alpha=self.alpha,
            gamma=self.gamma if gamma is None else gamma,
            reduction=self.reduction,
        )

    def forward(
        self,
        loc_preds: torch.Tensor,
        loc_targets: torch.Tensor,
        cls_preds: torch.Tensor,
        cls_targets: torch.Tensor,
    ):
        """
        Computes the Focal Loss.

        Args:
            loc_preds (torch.Tensor): Predictions for location, shape [batch_size, #anchors, 4].
            loc_targets (torch.Tensor): Encoded target for location, shape [batch_size, #anchors, 4].
            cls_preds (torch.Tensor): Predictions for class, shape [batch_size, #anchors, #classes].
            cls_targets (torch.Tensor): Encoded target for class, shape [batch_size, #anchors].

        Returns:
            dict: Dictionary containing losses for location, class, and total loss.
        """
        loss_dict = {}

        # Determine the number of positive anchor points in the batch.
        num_pos_anchors = (cls_targets > 0).sum().float()

        # Localization loss.
        pos_mask = cls_targets[:, :, 0] > 0
        loc_preds_pos = loc_preds[pos_mask].view(-1, 4)
        loc_targets_pos = loc_targets[pos_mask].view(-1, 4)
        loc_loss = (
            F.smooth_l1_loss(loc_preds_pos, loc_targets_pos, reduction="sum")
            / num_pos_anchors
        )
        loss_dict["loc_loss"] = loc_loss.item()

        # Class loss.
        total_loss = loc_loss
        pos_neg: torch.Tensor = cls_targets[:, :, 0] > -1
        for i, cls_pred in enumerate(cls_preds):
            mask = pos_neg.unsqueeze(2).expand_as(cls_pred)
            cls_loss = (
                self.focal_loss(
                    cls_pred[mask].view(-1, self.num_classes[i]),
                    cls_targets[:, :, i][pos_neg],
                    self.num_classes[i],
                    gamma=None if i == 0 else self.gamma,
                )
                / num_pos_anchors
            )
            loss_dict[f"cls_loss_{i}"] = cls_loss.item()
            total_loss += cls_loss

        # Total loss.
        loss_dict["loss"] = total_loss

        return loss_dict
