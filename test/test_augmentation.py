import numpy as np
from imgaug import BoundingBox

from src.augmentation import Augmenter, decode_bboxes, encode_bboxes
from src.config import Config


def test_decode_bboxes():
    bboxes = np.array([[10, 20, 30, 40, 1, 1], [50, 60, 70, 80, 2, 1]])
    bbs = decode_bboxes(bboxes)
    assert len(bbs) == 2
    assert isinstance(bbs[0], BoundingBox)
    assert bbs[0].x1 == 10.5
    assert bbs[0].y1 == 20.5
    assert bbs[0].x2 == 30.5
    assert bbs[0].y2 == 40.5
    assert bbs[1].label == 1


def test_encode_bboxes():
    bboxes = np.zeros((2, 6))
    bboxes[0, 4] = 1
    bboxes[0, 5] = 1
    bboxes[1, 4] = 2
    bboxes[1, 5] = 1
    bbs = [
        BoundingBox(x1=11, y1=21, x2=31, y2=41, label=0),
        BoundingBox(x1=51, y1=61, x2=71, y2=81, label=1),
    ]
    bboxes = encode_bboxes(bboxes, bbs)
    assert bboxes.shape == (2, 6)
    assert bboxes[0, 0] == 10.5
    assert bboxes[0, 1] == 20.5
    assert bboxes[0, 2] == 30.5
    assert bboxes[0, 3] == 40.5
    assert bboxes[0, 4] == 1
    assert bboxes[0, 5] == 1
    assert bboxes[1, 0] == 50.5
    assert bboxes[1, 1] == 60.5
    assert bboxes[1, 2] == 70.5
    assert bboxes[1, 3] == 80.5
    assert bboxes[1, 4] == 2
    assert bboxes[1, 5] == 1


def test_augmenter():
    cf = Config()
    augmenter = Augmenter(cf)
    bboxes = np.array([[10, 20, 30, 40, 1, 1], [50, 60, 70, 80, 2, 1]])
    data = {"dcm": np.zeros((512, 512)).astype("float16"), "annot": bboxes}
    output = augmenter(data)
    assert output["dcm"].shape[0] == cf.aug["image_size"]
    assert output["dcm"].shape[1] == cf.aug["image_size"]
    assert output["annot"].shape == (2, 6)
    assert output["annot"][0, 4] == 1
    assert output["annot"][0, 5] == 1
    assert output["annot"][1, 4] == 2
    assert output["annot"][0, 5] == 1
