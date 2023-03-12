from typing import List, Dict, Union

import imgaug.augmenters as iaa
import numpy as np
from imgaug import BoundingBox

from adapive_iou_retinanet.src.config import Config


def decode_bboxes(bboxes: np.ndarray) -> List:
    """
    Decodes a list of bounding boxes to a list of imgaug BoundingBox objects.

    Args:
    - bboxes (numpy.ndarray): A 2D array of shape (N, 6), where N is the number of bounding boxes
    and each row corresponds to a bounding box in the format (x1, y1, x2, y2, join_nr, ra_score).

    Returns:
    - List[imgaug.BoundingBox]: A list of imgaug BoundingBox objects.
    """
    bbs = [
        BoundingBox(
            x1=bbox[0] + 0.5,
            y1=bbox[1] + 0.5,
            x2=bbox[2] + 0.5,
            y2=bbox[3] + 0.5,
            label=label,
        )
        for label, bbox in enumerate(bboxes.tolist())
    ]
    return bbs


def encode_bboxes(bboxes: np.ndarray, bbs: List) -> np.ndarray:
    """
    Encodes a list of imgaug BoundingBox objects to a numpy array of bounding boxes.

    Args:
    - bboxes (numpy.ndarray):  A 2D array of shape (N, 6), where N is the number of bounding boxes
    and each row corresponds to a bounding box in the format (x1, y1, x2, y2, join_nr, ra_score).
    - bbs (List[imgaug.BoundingBox]): A list of imgaug BoundingBox objects.

    Returns:
    - numpy.ndarray: A 2D array of shape (N, 4), where N is the number of bounding boxes and each
    row corresponds to a bounding box in the format (x1, y1, x2, y2, join_nr, ra_score).
    """
    for i, b in enumerate(bbs):
        bboxes[i, :4] = np.array([b.x1 - 0.5, b.y1 - 0.5, b.x2 - 0.5, b.y2 - 0.5])
    return bboxes


SOMETIMES = lambda aug: iaa.Sometimes(0.5, aug)


class Augmenter:
    def __init__(self, cf: Config, predict_mode: bool = False) -> None:
        """
        Initializes an image augmenter object.

        Args:
        - cf (dict): Configuration file - see config.py
        - predict_mode (bool): If True, only resizing will be performed.
        """
        self.predict_mode = predict_mode

        # Initial resizing of images
        self.init_resize = iaa.Sequential(
            [iaa.Resize(size=(cf.aug["init_resize"], cf.aug["init_resize"]))]
        )
        self.image_size = cf.aug["image_size"]

        self.resize = iaa.Sequential(
            [iaa.Resize(size=(self.image_size, self.image_size))]
        )

        # Image transformations to apply
        # Use lambda function to generate a random sequence of augmentations
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Use iaa.Sequential to chain the image augmentations
        self.transform = iaa.Sequential(
            [
                iaa.GammaContrast(cf.aug["gamma_contrast"]),  # Adjust brightness
                iaa.Dropout(cf.aug["dropout"]),  # Random pixel dropout
                iaa.Flipud(cf.aug["flipud"]),  # Randomly flip the image vertically
                sometimes(
                    iaa.Rot90(cf.aug["rot90"])
                ),  # Randomly rotate the image by 90 degrees
                sometimes(
                    iaa.TranslateX(percent=cf.aug["translateX"])
                ),  # Randomly translate the image horizontally
                sometimes(
                    iaa.TranslateY(percent=cf.aug["translateY"])
                ),  # Randomly translate the image vertically
                iaa.ScaleX(
                    scale=cf.aug["scaleX"]
                ),  # Randomly scale the image horizontally
                iaa.ScaleY(
                    scale=cf.aug["scaleY"]
                ),  # Randomly scale the image vertically
                sometimes(
                    iaa.Crop(
                        px=(0, round(cf.aug["crop"] * cf.aug["init_resize"])),
                        keep_size=False,
                    )
                ),
                # Randomly crop the image
                iaa.Rotate(
                    cf.aug["rotate"]
                ),  # Randomly rotate the image by a specified degree
            ]
        )

    def __call__(
        self, data: Dict[str, Union[np.ndarray, List]]
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Applies augmentations to an image and its corresponding bounding boxes.

        Args:
        - data (dict): Dictionary containing two keys: 'dcm' for the image and 'annot' for the
        list of bounding boxes.

        Returns:
        - dict: Dictionary containing the augmented image and its corresponding bounding boxes.
        """
        img = data["dcm"]
        bboxes = data["annot"]
        bbs = decode_bboxes(bboxes)

        # Resize the image to an initial size
        img, bbs = self.init_resize(images=img, bounding_boxes=bbs)
        img[img <= 0] = 0

        # Apply image transformations (if not in prediction mode)
        if not self.predict_mode:
            img, bbs = self.transform(images=img, bounding_boxes=bbs)

        # Resize the image to the desired size
        img_aug, bbs_aug = self.resize(images=img, bounding_boxes=bbs)
        if isinstance(img_aug, list):
            img_aug = img_aug[0]

        # Update the bounding boxes
        bboxes = encode_bboxes(bboxes, bbs_aug)

        return {"dcm": img_aug, "annot": bboxes}
