"""Some helper functions for PyTorch."""
import math

import torch
import torch.nn as nn


def get_mean_and_std(dataset, max_load=10000):
    """Compute the mean and std value of dataset."""
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    N = min(max_load, len(dataset))
    for i in range(N):
        print(i)
        im, _, _ = dataset.load(1)
        for j in range(3):
            mean[j] += im[:, j, :, :].mean()
            std[j] += im[:, j, :, :].std()
    mean.div_(N)
    std.div_(N)
    return mean, std


def mask_select(input, mask, dim=0):
    """Select tensor rows/cols using a mask tensor.
    Args:
      input: (tensor) input tensor, sized [N,M].
      mask: (tensor) mask tensor, sized [N,] or [M,].
      dim: (tensor) mask dim.
    Returns:
      (tensor) selected rows/cols.
    Example:
    >>> a = torch.randn(4,2)
    >>> a
    -0.3462 -0.6930
     0.4560 -0.7459
    -0.1289 -0.9955
     1.7454  1.9787
    [torch.FloatTensor of size 4x2]
    >>> i = a[:,0] > 0
    >>> i
    0
    1
    0
    1
    [torch.ByteTensor of size 4]
    >>> masked_select(a, i, 0)
    0.4560 -0.7459
    1.7454  1.9787
    [torch.FloatTensor of size 2x2]
    """
    index = mask.nonzero().squeeze(1)
    return input.index_select(dim, index)


def meshgrid(x, y, row_major=True):
    """Return meshgrid in range x & y.
    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.
    Returns:
      (tensor) meshgrid, sized [x*y,2]
    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]
    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    """
    a = torch.arange(0, x)
    b = torch.arange(0, y)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1) if row_major else torch.cat([yy, xx], 1)


def change_box_order(boxes, order):
    """Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).
    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.
    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    """
    assert order in ["xyxy2xywh", "xywh2xyxy"]
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == "xyxy2xywh":
        return torch.cat([(a + b) / 2, b - a + 1], 1)
    b -= 1
    return torch.cat([a - b / 2, a + b / 2], 1)


def box_iou(box1, box2, order="xyxy", only_inter: bool = False):
    """Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.
    Return:
      (tensor) iou, sized [N,M].
    """
    if order == "xywh":
        box1 = change_box_order(box1, "xywh2xyxy")
        box2 = change_box_order(box2, "xywh2xyxy")

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt + 1).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)  # [N,]
    area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)  # [M,]
    if only_inter:
        iou = inter / (area1[:, None])
    else:
        iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      pick: (tensor) selected indices.
    """
    # initialize the list of picked indexes
    pick = []

    # if there are no boxes, return empty list
    if bboxes.shape == torch.Size([0, 4]):
        return pick

    # grab the coordinates of the bounding boxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by score
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending=True)

    while order.numel() > 0:

        i = order[0] if order.numel() != 1 else order
        pick.append(i)

        if order.numel() == 1:
            break

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        # compute the width and height of the bounding box
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)

        # compute the ratio of overlap
        inter = w * h
        ovr = inter / areas[order[1:]]

        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break

        # delete all indexes from the index list that have
        order = order[ids + 1]

    # return only the inds of boxes that were picked
    return torch.LongTensor(pick)


def softmax(x):
    """Softmax along a specific dimension.
    Args:
      x: (tensor) input tensor, sized [N,D].
    Returns:
      (tensor) softmaxed tensor, sized [N,D].
    """
    xmax, _ = x.max(1)
    x_shift = x - xmax.view(-1, 1)
    x_exp = x_shift.exp()
    return x_exp / x_exp.sum(1).view(-1, 1)


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    labels[labels == -1] = 0
    y = torch.eye(num_classes, device=labels.device).long()# [D,D]
    return y[labels.long()]  # [N,D]


def msr_init(net):
    """Initialize layer parameters."""
    for layer in net:
        if type(layer) == nn.Conv2d:
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2.0 / n))
            layer.bias.data.zero_()
        elif type(layer) == nn.BatchNorm2d:
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif type(layer) == nn.Linear:
            layer.bias.data.zero_()
