"""Encode object boxes and labels."""
import math

import torch

from adapive_iou_retinanet.src.utils import meshgrid, box_iou, box_nms, change_box_order


import torch
import math


class DataEncoder:
    def __init__(self, cf):
        # cf is a config object that contains pyramid levels
        self.fpn_layer = cf.pyramid_levels

        # initialize anchor areas for different pyramid levels
        self.anchor_areas = [
            4.0 * 4.0,
            8.0 * 8.0,
            16.0 * 16.0,
            32 * 32.0,
            64 * 64.0,
            128 * 128.0,
            256.0 * 256.0,
        ]  # p1 -> p7

        # define aspect ratios and scale ratios for anchors
        self.aspect_ratios = [1 / 2.0, 1 / 1.0, 2 / 1.0]
        self.scale_ratios = [1.0, pow(2, 1 / 3.0), pow(2, 2 / 3.0)]

        # update anchor areas based on the pyramid levels
        self._update_anchor_areas()

        # calculate anchor width and height for each feature map
        self.anchor_wh = self._get_anchor_wh()

    def _update_anchor_areas(self):
        # update anchor areas based on the pyramid levels
        new_areas = []
        for i in self.fpn_layer:
            new_areas.append(self.anchor_areas[i - 1])
        self.anchor_areas = new_areas

    def _get_anchor_wh(self):
        """Compute anchor width and height for each feature map.
        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        """
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # iterate over aspect ratios
                # calculate height and width of anchor box
                h = math.sqrt(s / ar)
                w = ar * h
                for sr in self.scale_ratios:  # iterate over scale ratios
                    # scale anchor box
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        # return anchor width and height for each feature map
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        """Compute anchor boxes for each feature map.
        Args:
          input_size: (tensor) model input size of (w,h).
        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        """
        num_fms = len(self.anchor_areas)
        fm_sizes = [
            (input_size / pow(2.0, i + min(self.fpn_layer))).ceil()
            for i in range(num_fms)
        ]  # p2 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5  # [fm_h*fm_w, 2]
            xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 9, 2)
            wh = self.anchor_wh[i].view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2)
            box = torch.cat([xy, wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size, pos=0.5, neg=0.4):
        """Encode target bounding boxes and class labels.
        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)
        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).
        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        """
        input_size = (
            torch.Tensor([input_size, input_size])
            if isinstance(input_size, int)
            else torch.Tensor(input_size)
        )
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, "xyxy2xywh")

        ious = box_iou(anchor_boxes, boxes, order="xywh")
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = labels[max_ids]

        cls_targets[max_ious < pos] = 0
        ignore = (max_ious > neg) & (max_ious <= pos)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1  # for now just mark ignored to -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        """Decode outputs back to bouding box locations and class labels.
        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).
        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        """
        CLS_THRESH = 0.2
        NMS_THRESH = 0.1

        input_size = (
            torch.Tensor([input_size, input_size])
            if isinstance(input_size, int)
            else torch.Tensor(input_size)
        )
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)  # [#anchors,4]

        # score, labels = cls_preds.float().softmax(1).max(1)
        if cls_preds.shape.__len__() == 2:
            score, labels = cls_preds.sigmoid().max(1)  # [#anchors,]
        else:
            score, labels = cls_preds.sigmoid()

        score = score[labels != 0]
        boxes = boxes[labels != 0]
        labels = labels[labels != 0]

        ids = score > CLS_THRESH
        # if len(ids) > 3000:
        #    ii, kk = score.sort()
        #    ids[kk > 3000] = False
        ids = ids.nonzero().squeeze()  # [#obj,]
        if ids.shape == torch.Size([]):
            ids = torch.tensor([ids])
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
        return boxes[ids][keep], labels[ids][keep]

    def multi_decode(self, loc_preds, multi_cls_preds, input_size):
        """Decode outputs back to bouding box locations and class labels.
        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          multi_cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).
        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        """
        cls_preds = multi_cls_preds[0]
        CLS_THRESH = 0.01
        NMS_THRESH = 0.9

        input_size = (
            torch.Tensor([input_size, input_size])
            if isinstance(input_size, int)
            else torch.Tensor(input_size)
        )
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)  # [#anchors,4]

        # score, labels = cls_preds.float().softmax(1).max(1)
        if cls_preds.shape.__len__() == 2:
            score, labels = cls_preds.sigmoid().max(1)  # [#anchors,]
        else:
            score, labels = cls_preds.sigmoid()

        background = labels == 0
        score = score[background == False]
        boxes = boxes[background == False]
        labels = labels[background == False]

        ids = score > CLS_THRESH
        ids = ids.nonzero().squeeze()  # [#obj,]
        if ids.shape == torch.Size([]):
            ids = torch.tensor([ids])
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)

        scores = [labels[ids][keep]]
        for i in range(1, len(multi_cls_preds)):
            cls_preds = multi_cls_preds[i]
            cls_preds = cls_preds[:, 1:]  # remove background
            _, labels = cls_preds.sigmoid().max(1)
            labels = labels[background == False]
            scores.append(labels[ids][keep])

        return boxes[ids][keep], scores
