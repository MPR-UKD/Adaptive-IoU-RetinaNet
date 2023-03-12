import os
import os.path
import shutil

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List
import pytorch_lightning as pl
import torch
from adapive_iou_retinanet.src.config import Config
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score

from adapive_iou_retinanet.src.encoder import DataEncoder
from adapive_iou_retinanet.src.fpn import RetinaFPN50, RetinaFPN101
from adapive_iou_retinanet.src.loss import FocalLoss
from adapive_iou_retinanet.src.utils import one_hot_embedding
from sklearn.metrics import average_precision_score


class Header(nn.Module):
    """
    A module that creates a header for the RetinaNet network.
    This header consists of several convolutional layers with
    optional activation and normalization functions.

    Args:
        out_planes (int): The number of output channels in the final layer.
        features (int, optional): The number of input and output channels in
            the convolutional layers (default=128).
        ks (int, optional): The size of the convolutional kernel (default=3).
        pad (int, optional): The amount of zero-padding in the convolution (default=1).
        stride (int, optional): The stride of the convolution (default=1).
        relu (str, optional): The type of activation function to use, either "relu" or "leakyrelu" (default=None).
        norm (str, optional): The type of normalization function to use, either "batchnorm" or "instancenorm" (default=None).
    """

    def __init__(
        self,
        out_planes: int,
        features: int = 128,
        ks: int = 3,
        pad: int = 1,
        stride: int = 1,
        relu: str = None,
        norm: str = None,
    ):
        super().__init__()
        self.layer = self._make_head(out_planes, features, ks, pad, stride, relu, norm)

    def _make_head(
        self,
        out_planes: int,
        features: int = 16,
        ks: int = 3,
        pad: int = 1,
        stride: int = 1,
        relu: str = None,
        norm: str = None,
    ) -> nn.Sequential:
        """
        A helper method that creates the convolutional layers for the header.

        Args:
            out_planes (int): The number of output channels in the final layer.
            features (int, optional): The number of input and output channels in
                the convolutional layers (default=128).
            ks (int, optional): The size of the convolutional kernel (default=3).
            pad (int, optional): The amount of zero-padding in the convolution (default=1).
            stride (int, optional): The stride of the convolution (default=1).
            relu (str, optional): The type of activation function to use, either "relu" or "leakyrelu" (default=None).
            norm (str, optional): The type of normalization function to use, either "batchnorm" or "instancenorm" (default=None).

        Returns:
            nn.Sequential: A sequential module containing the convolutional layers.
        """
        layers = []
        for _ in range(4):
            layers.append(
                nn.Conv2d(
                    features, features, kernel_size=ks, stride=stride, padding=pad
                )
            )
            if relu is not None:
                layers.append(nn.ReLU() if relu.lower() == "relu" else nn.LeakyReLU())
            if norm is not None:
                layers.append(
                    nn.BatchNorm2d(features)
                    if norm.lower() == "batchnorm"
                    else nn.InstanceNorm2d(features)
                )
        layers.append(
            nn.Conv2d(features, out_planes, kernel_size=ks, stride=stride, padding=pad)
        )
        return nn.Sequential(*layers)


class RegressionModel(Header):
    """
    A module that creates the regression branch of the RetinaNet network.
    This branch predicts the offsets for each anchor box.

    Args:
        out_planes (int): The number of output channels in the final layer.
        relu (str, optional): The type of activation function to use in the header, either "relu" or "leakyrelu" (default=None).
        norm (str, optional): The type of normalization function to use in the header, either "batchnorm" or "instancenorm" (default=None).
    """

    def __init__(self, out_planes: int, relu: str = None, norm: str = None):
        super().__init__(out_planes, relu=relu, norm=norm)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass of the regression branch.

        Args:
            img (torch.Tensor): A tensor of shape (N, C, H, W) representing the input image.

        Returns:
            torch.Tensor: A tensor of shape (N, H * W * n_anchors, 4) representing the predicted offsets for each anchor box.
        """
        out = self.layer(img)
        # [N, n_anchors * 4 H, W] -> [N, H, W, n_anchors * 4] -> [N, H * W * n_anchors, 4]
        return out.permute(0, 2, 3, 1).contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(Header):
    """
    A module that creates the classification branch of the RetinaNet network.
    This branch predicts the class probabilities for each anchor box.

    Args:
        out_planes (int): The number of output channels in the final layer.
        num_cls (int): The number of object classes to predict.
        relu (str, optional): The type of activation function to use in the header, either "relu" or "leakyrelu" (default=None).
        norm (str, optional): The type of normalization function to use in the header, either "batchnorm" or "instancenorm" (default=None).
    """

    def __init__(
        self, out_planes: int, num_cls: int, relu: str = None, norm: str = None
    ):
        super().__init__(out_planes, relu=relu, norm=norm)
        self.num_cls = num_cls

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass of the classification branch.

        Args:
            img (torch.Tensor): A tensor of shape (N, C, H, W) representing the input image.

        Returns:
            torch.Tensor: A tensor of shape (N, H * W * n_anchors, n_classes) representing the predicted class probabilities for each anchor box.
        """
        out = self.layer(img)
        # [N, n_anchors * n_classes, H, W] -> [N, H, W, n_anchors * n_classes] -> [N, H * W * n_anchors, n_classes]
        return out.permute(0, 2, 3, 1).contiguous().view(out.shape[0], -1, self.num_cls)


tensor = lambda x: torch.tensor(x).float()


class RetinaNet(pl.LightningModule):
    def __init__(self, n_cls: List[int], epoch_dict: dict, cf: Config):
        super(RetinaNet, self).__init__()
        self.fpn = (
            RetinaFPN50() if cf.res_architecture == "resnet50" else RetinaFPN101()
        )
        self.num_anchors = 9
        self.loc_head = RegressionModel(
            out_planes=self.num_anchors * 4, relu=cf.relu, norm=cf.norm
        )
        self.learning_rate = cf.learning_rate
        self.weight_decay = cf.weight_decay
        if type(n_cls) is not list:
            n_cls = [n_cls]

        self.cls_head1 = ClassificationModel(
            out_planes=n_cls[0] * self.num_anchors,
            num_cls=n_cls[0],
            relu=cf.relu,
            norm=cf.norm,
        )
        if len(n_cls) >= 2:
            self.cls_head2 = ClassificationModel(
                out_planes=n_cls[1] * self.num_anchors,
                num_cls=n_cls[1],
                relu=cf.relu,
                norm=cf.norm,
            )

        self.criterion = FocalLoss(num_classes=n_cls)
        self.encoder = DataEncoder(cf=cf)
        self.automatic_optimization = True
        self.n_cls = n_cls
        self.log_step_size = cf.log_step_size
        self.img_step_size = cf.img_step_size
        self.init_logs()
        self.epoch_dict = epoch_dict

    def on_epoch_start(self):
        self.epoch_dict["epoch"] = self.current_epoch

    def init_logs(self):
        self.logs = {"val_join_accuray": [], "val_join_erosion": []}

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.tensor([o["loss"] for o in outputs]).mean()
        self.log("val_loss", loss)
        val_logs = self.logs["val_join_accuray"]
        if len(val_logs) == 0:
            self.log("Val_Accuracy", tensor(0))
            self.log("Erosion_Accuracy", tensor(0))
            self.log(f"mAP_0", tensor(0))
            self.log(f"mAP_1", tensor(0))
            self.init_logs()
            return
        for ii, key in enumerate(["val_join_accuray", "val_join_erosion"]):
            val_logs = self.logs[key]
            y_true = []
            y_pred = []
            for val_log in val_logs:
                for v in val_log:
                    y_true.append(v[0])
                    y_pred.append(v[1])
                df = pd.DataFrame(
                    data=confusion_matrix(
                        y_true, y_pred, labels=[i for i in range(self.n_cls[ii])]
                    ),
                    columns=[i for i in range(self.n_cls[ii])],
                    index=[i for i in range(self.n_cls[ii])],
                )
            mAP = calc_mAP(self.n_cls[ii], y_true, y_pred)
            self.log(f"mAP_{ii}", tensor(mAP))
            if self.current_epoch % self.img_step_size == 0:
                df.to_csv(
                    f".{os.sep}images_bbox{os.sep}{self.current_epoch}{os.sep}{key}summary_of_predictions.csv"
                )
                print(f"{key}: {accuracy_score(y_true, y_pred)}")
                for key in outputs[0].keys():
                    print(f"{key}: {torch.tensor([o[key] for o in outputs]).mean()}")
                print(f"mAP_{ii}", mAP)
            if ii == 0:
                self.log("Val_Accuracy", tensor(accuracy_score(y_true, y_pred)))

            else:
                self.log("Erosion_Accuracy", tensor(accuracy_score(y_true, y_pred)))

        self.init_logs()

    def validation_step(self, batch, batch_idx):
        img, annont = batch["dcm"], batch["annot"]
        loc_preds, cls_preds = self(img)
        loss_dict = self.criterion(
            loc_preds, annont[:, 0, :, :4], cls_preds, annont[:, 0, :, 4:]
        )
        loc_preds = loc_preds.cpu()

        # if self.current_epoch % self.log_step_size != 0:
        #    return loss_dict
        if self.current_epoch % self.img_step_size == 0:
            draw = True if batch_idx == 0 else False
        else:
            draw = False

        if draw:
            img = img.cpu().numpy()
            if not os.path.isdir(f".{os.sep}images_bbox"):
                os.mkdir(f".{os.sep}images_bbox")
            if os.path.isdir(f".{os.sep}images_bbox{os.sep}{self.current_epoch}"):
                shutil.rmtree(f".{os.sep}images_bbox{os.sep}{self.current_epoch}")
            os.mkdir(f".{os.sep}images_bbox{os.sep}{self.current_epoch}")
        for b in range(loc_preds.shape[0]):
            t = [
                one_hot_embedding(annont[b, 0, :, 4 + i].cpu(), self.n_cls[i])
                for i in range(len(self.n_cls))
            ]
            targets = get_boxes(
                self.encoder, annont[b, 0, :, :4], t, tuple(img.shape[-2:])
            )
            c = [cls_preds[i][b].cpu() for i in range(len(self.n_cls))]
            predictions = get_boxes(
                self.encoder, loc_preds[b], c, tuple(img.shape[-2:])
            )

            if draw:
                """bbox"""
                # Create figure and axes
                fig, ax = plt.subplots()
                # Display the image
                ax.imshow(img[b, 0], cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

                if self.current_epoch == 0:

                    ax = draw_on_fig(ax, targets[0], targets[1], "b")
                else:
                    ax = draw_on_fig(ax, predictions[0], predictions[1], "r")
                plt.savefig(
                    f".{os.sep}images_bbox{os.sep}{self.current_epoch}{os.sep}test_{b}.png"
                )
                plt.close()
            if self.current_epoch > 0:
                try:
                    self.logs["val_join_accuray"].append(
                        sort_pred_and_targets(targets, predictions, cls_num=0)
                    )
                except:
                    self.logs["val_join_accuray"].append([])
                try:
                    self.logs["val_join_erosion"].append(
                        sort_pred_and_targets(targets, predictions, cls_num=1)
                    )
                except:
                    self.logs["val_join_erosion"].append([])
        return loss_dict

    def set_name(self, name):
        self.name = name

    def predict_step(self, batch, dataloader_idx):
        img, annont, org_img = batch["dcm"], batch["annot"], batch["dcm_file"]
        return self.predict(img, annont, org_img)

    def training_epoch_end(self, outputs):
        mean_loss = torch.tensor([o["loss"] for o in outputs]).mean()
        self.log("loss", mean_loss)

    def training_step(self, batch, batch_idx):
        img, annont = batch["dcm"], batch["annot"]

        loc_preds, cls_preds = self(img)
        loss_dict = self.criterion(
            loc_preds, annont[:, 0, :, :4], cls_preds, annont[:, 0, :, 4:]
        )
        return loss_dict

    def forward(self, img):
        fms = self.fpn(img)
        loc_preds = []
        cls_preds1 = []
        cls_preds2 = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            loc_preds.append(loc_pred)

            cls_pred1 = self.cls_head1(fm)
            cls_preds1.append(cls_pred1)

            if len(self.n_cls) >= 2:
                cls_pred2 = self.cls_head2(fm)
                cls_preds2.append(cls_pred2)

        out_loc = torch.cat(loc_preds, 1)
        out_cls = [torch.cat(cls_preds1, 1)]
        if len(cls_preds2) > 0:
            out_cls.append(torch.cat(cls_preds2, 1))

        return out_loc, out_cls

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=10**-6
        )

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=20,
                cooldown=40,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] <= bb1["x2"]
    assert bb1["y1"] <= bb1["y2"]
    assert bb2["x1"] <= bb2["x2"]
    assert bb2["y1"] <= bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    if iou.isnan():
        iou = 0.0
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_boxes(encoder, loc_preds, cls_preds, shape):
    boxes, targets = encoder.multi_decode(loc_preds.cpu(), cls_preds, shape)
    if boxes.size() == torch.Size([1, 1, 4]):
        boxes = boxes[0]
    boxes, targets = clear_detections(boxes, targets)
    return boxes, targets


def draw_on_fig(ax, boxes, targets, color):
    if boxes is None:
        return ax
    for i, coord in enumerate(boxes):
        label = str(int(targets[i][0].item())) + "_" + str(int(targets[i][1].item()))
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (coord[0], coord[1]),
            coord[2] - coord[0],
            coord[3] - coord[1],
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(coord[0], coord[1], label, color=color, fontsize="small")  # 'xx-small')
    return ax


def sort_pred_and_targets(targets, predictions, cls_num=0):
    accuracy = []
    for i, coords in enumerate(targets[0]):
        target_cls = int(targets[1][i][cls_num].item())
        target_join = int(targets[1][i][0].item())
        best_iou = 0.0
        pred_cls = None
        for j, pred_coords in enumerate(predictions[0]):
            iou = get_iou(
                {"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]},
                {
                    "x1": pred_coords[0],
                    "y1": pred_coords[1],
                    "x2": pred_coords[2],
                    "y2": pred_coords[3],
                },
            )
            if iou > best_iou:
                best_iou = iou
                pred_cls = int(predictions[1][j][cls_num].item())
                pred_join = int(predictions[1][j][0].item())
        if cls_num != 0:
            if pred_cls is not None:
                if pred_join != target_join:
                    pred_cls = -1
        if best_iou < 0.3:
            pred_cls = -1
        accuracy.append((target_cls, pred_cls))
    return accuracy


def clear_detections(boxes, targets):
    joins = targets[0]
    boxes_out, targets_out = [], []

    for i in range(33):
        xs, ys, size_xs, size_ys, scores = [], [], [], [], []
        count = 0
        for b, s in zip(boxes, list(joins)):
            if s == i:
                xs.append(abs((b[0] - b[2])) / 2 + b[0])
                ys.append(abs((b[1] - b[3])) / 2 + b[1])
                size_xs.append(abs((b[2] - b[0])))
                size_ys.append(abs((b[3] - b[1])))
                scores.append([targets[i][count] for i in range(len(targets))])
            count += 1
        if len(xs) > 0:
            x = np.mean(xs)
            y = np.mean(ys)
            size_x = np.mean(size_xs) / 2
            size_y = np.mean(size_ys) / 2
            boxes_out.append(np.array([x - size_x, y - size_y, x + size_x, y + size_y]))
            targets_out.append(np.median(scores, axis=0))

    return torch.tensor(boxes_out), torch.tensor(targets_out).view(
        len(boxes_out), len(targets)
    )


def calc_mAP(n_cls, y_true, y_pred):
    mAP = 0
    for k in range(n_cls - 1):
        t, y = [], []
        for iii in range(len(y_true)):
            if y_true[iii] == k:
                t.append(1)
            else:
                t.append(0)
            if y_pred[iii] == k:
                y.append(1)
            else:
                y.append(0)
        if np.array(y).max() == 0 and np.array(t).max() == 0:
            n_cls -= 1
        elif np.array(y).max() == 0 and np.array(t).max() != 0:
            mAP += 0.0
        elif np.array(y).max() != 0 and np.array(t).max() == 0:
            mAP += 0.0
        else:
            mAP += average_precision_score(t, y)
    return mAP / n_cls


def calc_mean_iou(targets, predictions, cls_num=0):
    mean_iou = []
    for i, coords in enumerate(targets[0]):
        target_cls = int(targets[1][i][cls_num].item())
        target_join = int(targets[1][i][0].item())
        best_iou = 0.0
        pred_cls = None
        for j, pred_coords in enumerate(predictions[0]):
            pred_coords[torch.isnan(pred_coords)] = 0.0
            iou = get_iou(
                {"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]},
                {
                    "x1": pred_coords[0],
                    "y1": pred_coords[1],
                    "x2": pred_coords[2],
                    "y2": pred_coords[3],
                },
            )
            if iou > best_iou:
                best_iou = iou
                pred_cls = int(predictions[1][j][cls_num].item())
                pred_join = int(predictions[1][j][0].item())
        if cls_num != 0:
            if pred_cls is not None:
                if pred_join != target_join:
                    best_iou = 0

        mean_iou.append(best_iou)
    return mean_iou
