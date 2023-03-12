import numpy as np
from imgaug import BoundingBox

from ..src.encoder import DataEncoder
from ..src.config import Config
import torch


def test_encode():
    # create a DataEncoder object with a Config object
    data_encoder = DataEncoder(cf=Config())

    # create a tensor of bounding boxes
    bboxes = torch.tensor(
        [
            [54, 327, 100, 344, 1, 2],
            [75, 325, 141, 348, 2, 1],
            [75, 325, 118, 346, 3, 1],
        ]
    )

    # encode the bounding boxes using the encode method with input_size=(300, 300)
    encoded_boxes1, _ = data_encoder.encode(
        boxes=bboxes[:, :4],
        labels=bboxes[:, 4],
        input_size=(300, 300),
    )

    # encode the bounding boxes again using the encode method with input_size=(300, 300)
    encoded_boxes2, _ = data_encoder.encode(
        boxes=bboxes[:, :4],
        labels=bboxes[:, 4],
        input_size=(300, 300),
    )

    # check if the two encoded outputs are equal using an assertion statement
    assert torch.allclose(encoded_boxes1, encoded_boxes2, atol=1e-5)


def test_encode_pyramid_levels_3_4():
    cf = Config()

    # create a DataEncoder object with pyramid_levels = [3, 4]
    data_encoder = DataEncoder(cf=cf)

    # create a tensor of bounding boxes
    bboxes = torch.tensor(
        [
            [54, 327, 100, 344, 1, 2],
            [75, 325, 141, 348, 2, 1],
            [75, 325, 118, 346, 3, 1],
        ]
    )

    # encode the bounding boxes using the encode method with input_size=(300, 300)
    encoded_boxes1, _ = data_encoder.encode(
        boxes=bboxes[:, :4],
        labels=bboxes[:, 4],
        input_size=(300, 300),
    )

    assert encoded_boxes1.shape[1] == 4
    assert encoded_boxes1.shape[0] == 16245


def test_encode_pyramid_levels_3_4_5():
    cf = Config()
    cf.pyramid_levels = [3, 4, 5]

    # create a DataEncoder object with pyramid_levels = [3, 4, 5]
    data_encoder = DataEncoder(cf=cf)

    # create a tensor of bounding boxes
    bboxes = torch.tensor(
        [
            [54, 327, 100, 344, 1, 2],
            [75, 325, 141, 348, 2, 1],
            [75, 325, 118, 346, 3, 1],
        ]
    )

    # encode the bounding boxes using the encode method with input_size=(300, 300)
    encoded_boxes1, _ = data_encoder.encode(
        boxes=bboxes[:, :4],
        labels=bboxes[:, 4],
        input_size=(300, 300),
    )

    assert encoded_boxes1.shape[1] == 4
    assert encoded_boxes1.shape[0] == 17145


def test_encode_decode():
    pass
