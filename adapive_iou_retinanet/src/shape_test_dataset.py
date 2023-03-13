import random
from PIL import Image, ImageDraw
from adapive_iou_retinanet.src.encoder import DataEncoder
from typing import Callable, List, Dict, Union
from adapive_iou_retinanet.src.config import Config
import numpy as np
from copy import deepcopy
import torch
from adapive_iou_retinanet.src.dataset import Dataset


class TestDataset(Dataset):
    def __init__(
        self,
        transformer: Callable,
        number_of_images: int,
        config: Config,
        pos: float = 0.5,
        neg: float = 0.3,
        adaptive_epochs: int = 0,
        epoch_dict: dict = None,
    ):
        self.transform = transformer
        self.number_of_images = number_of_images
        self.encoder = DataEncoder(cf=config)
        self.epoch = epoch_dict
        self.pos = pos
        self.neg = neg
        self.adaptive_epochs = adaptive_epochs
        self.data = self.load_data()

    def load_data(self):
        shapes = Shapes(size=1000, border=0)
        data = []
        for i in range(self.number_of_images):
            img, bboxes = shapes.draw()
            data.append(
                {"bboxes": bboxes[:, :4], "image": img, "scores": bboxes[:, 4:]}
            )
        return data

    def num_classes(self) -> List:
        return [5, 3]

    def __len__(self) -> int:
        """
        Get the total number of samples in the input data.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def load_annotations(self, idx: int) -> np.ndarray:
        """
        Load the annotations for the sample at the specified index.

        Args:
            idx (int): The index of the sample.

        Returns:
            np.ndarray: An array containing the annotations for the sample.
        """
        data = self.data[idx]
        bbox_list = np.array(deepcopy(data["bboxes"]))
        score_list = data["scores"]
        annotations = np.zeros((0, 6))
        for idx in range(len(score_list)):
            annotation = np.zeros((1, 6))
            annotation[0, :4] = bbox_list[idx, :]
            annotation[0, 4:] = np.array(score_list[idx])
            annotations = np.append(annotations, annotation, axis=0)
        return annotations

    def transform_annotations(self, annotations: np.ndarray, size: int) -> torch.Tensor:
        """
        Transform the annotations for the input sample.

        Args:
            annotations (np.ndarray): A 2D numpy array containing the annotations for the sample.
            size (int): The input size for the transformation.

        Returns:
            torch.Tensor: A tensor containing the transformed annotations for the sample.
        """

        pos_ratio, neg_ratio = self.adaptive_iou()

        # Encode the annotations using the provided encoder
        boxes, targets_1 = self.encoder.encode(
            boxes=annotations[:, :4],
            labels=annotations[:, 4],
            input_size=size,
            pos=pos_ratio,
            neg=neg_ratio,
        )
        _, targets_2 = self.encoder.encode(
            boxes=annotations[:, :4],
            labels=annotations[:, 5],
            input_size=size,
            pos=pos_ratio,
            neg=neg_ratio,
        )

        # Store the transformed annotations in a list of lists
        transformed_annotations = [
            [
                boxes[i][0],
                boxes[i][1],
                boxes[i][2],
                boxes[i][3],
                targets_1[i],
                targets_2[i],
            ]
            for i in range(boxes.shape[0])
        ]

        # Convert the list of lists to a PyTorch tensor
        return torch.tensor(transformed_annotations)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Get the item at the specified index.

        Args:
            idx (int): The index of the item to get.

        Returns:
            Dict[str, Union[str, torch.Tensor]]: A dictionary containing the item data and metadata.
        """
        # Load the data and annotations for the specified index
        data = self.data[
            idx
        ]  # {"bboxes": bboxes[:4], "image": img, "scores": bboxes[4:]}
        dcm = data["image"]
        annotations = self.load_annotations(idx)

        data_dict = {
            "dcm": dcm.astype("float32"),
            "annot": annotations.astype("float32"),
        }

        # Apply the transformation function to the item data, if provided
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        # Add metadata to the item data dictionary

        data_dict["annot"] = self.transform_annotations(
            torch.from_numpy(data_dict["annot"]), data_dict["dcm"].shape
        )

        data_dict["annot"] = torch.tensor(np.expand_dims(data_dict["annot"], axis=0))
        data_dict["dcm"] = torch.tensor(
            np.expand_dims(data_dict["dcm"].astype("float32"), axis=0)
        )
        return data_dict


class Shapes:
    """
    A class that generates an image with three circles and three rectangles,
    draws a bounding box around each shape, and returns the resulting image
    and bounding box coordinates.
    """

    def __init__(self, size=256, border=10):
        """
        Initialize the Shapes object with the desired image size and border width.

        Args:
            size (int): The size of the square image to generate (default: 256).
            border (int): The width of the border around the image (default: 10).
        """
        self.size = size
        self.border = border

    def draw(self):
        """
        Draw an image with three circles and three rectangles, and draw a bounding
        box around each shape.

        Returns:
            PIL.Image.Image: The resulting image.
            list: A list of (x_min, y_min, x_max, y_max) coordinates for each shape.
        """
        # Initialize a new PIL image with a black background.
        img = Image.new("RGB", (self.size, self.size), (0, 0, 0))

        # Initialize a new PIL draw object to draw shapes on the image.
        draw = ImageDraw.Draw(img)

        # Generate the coordinates of the circles
        circles = [
            (
                random.randint(int(0.1 * self.size), int(0.9 * self.size)),
                random.randint(int(0.1 * self.size), int(0.9 * self.size)),
            )
            for _ in range(4)
        ]

        # Draw the circles on the image, and compute their bounding boxes.
        boxes = []
        rs = [(i + 1) * int(self.size / 20) for i in range(4)]
        score_1 = 1
        for (cx, cy), r in zip(circles, rs):
            if random.randint(0, 1) == 1:
                score_2 = 1
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(120, 120, 120))
            else:
                score_2 = 2
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255))
            r += 2
            boxes.append((cx - r, cy - r, cx + r, cy + r, score_1, score_2))
            score_1 += 1

        return np.array(img.convert("L")), np.array(boxes)
