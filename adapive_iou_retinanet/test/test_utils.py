from ..src import utils
import torch
import pytest


def test_meshgrid():
    meshgrid_coords = utils.meshgrid(2, 3)
    assert meshgrid_coords.shape == torch.Size([6, 2])
    assert (
        meshgrid_coords
        == torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]])
    ).all()


def test_change_box_order():
    xyxy = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
    xywh = torch.tensor([[20.0, 30.0, 21.0, 21.0], [60.0, 70.0, 21.0, 21.0]])
    assert (utils.change_box_order(xyxy, "xyxy2xywh") == xywh).all()
    assert (utils.change_box_order(xywh, "xywh2xyxy") == xyxy).all()


def test_box_iou():
    box1 = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
    box2 = torch.tensor([[20, 30, 40, 50], [60, 70, 80, 90], [100, 200, 300, 400]])
    assert torch.allclose(
        utils.box_iou(box1, box2, "xyxy"),
        torch.tensor([[0.1590, 0.0000, 0.0000], [0.0000, 0.1590, 0.0000]]),
        atol=1e-5,
    )


def test_one_hot_embedding():
    labels = torch.tensor([1, 2, 0, 0, 1])
    one_hot = utils.one_hot_embedding(labels, 3)
    one_hot_gt = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert torch.allclose(one_hot, one_hot_gt, atol=1e-5)


def test_box_nms():
    bboxes = torch.tensor([[10, 20, 30, 40], [11, 21, 30, 40], [50, 60, 70, 80]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert utils.box_nms(bboxes, scores, 0.3)[0] == 0
    assert utils.box_nms(bboxes, scores, 0.3)[1] == 2


def test_softmax():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    assert torch.allclose(
        utils.softmax(x),
        torch.tensor([[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]]),
        atol=1e-2,
    )


if __name__ == "__main__":
    pytest.main()
