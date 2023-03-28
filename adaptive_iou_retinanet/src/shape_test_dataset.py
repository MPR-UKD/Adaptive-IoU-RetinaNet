import random
from PIL import Image, ImageDraw
from adaptive_iou_retinanet.src.encoder import DataEncoder
from typing import Callable, List, Dict, Union
from adaptive_iou_retinanet.src.config import Config
import numpy as np
from copy import deepcopy
import torch
from adaptive_iou_retinanet.src.abstract_dataset import Dataset


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
        shapes = Shapes(size=800)
        data = []
        for i in range(self.number_of_images):
            img, bboxes = shapes.draw(i)
            data.append(
                {"bboxes": bboxes[:, :4], "image": img, "scores": bboxes[:, 4:]}
            )
        return data

    def num_classes(self) -> List:
        return [8, 5]

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

    def __init__(self, size=256):
        """
        Initialize the Shapes object with the desired image size and border width.

        Args:
            size (int): The size of the square image to generate (default: 256).
        """
        self.drawer = None
        self.size = size

    def draw_circle(self, x, y, r, low, upper):
        fill = tuple([random.randint(low, upper) for _ in range(3)])
        self.drawer.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def draw_ellipse(self, x, y, r, low, upper):
        fill = tuple([random.randint(low, upper) for _ in range(3)])
        self.drawer.ellipse((x - 2 * r, y - r, x + 2 * r, y + r), fill=fill)

    def draw_rect(self, x, y, r, low, upper):
        fill = tuple([random.randint(low, upper) for _ in range(3)])
        self.drawer.rectangle((x - r, y - r, x + r, y + r), fill=fill)

    def draw_line(self, x, y, r, low, upper):
        fill = tuple([random.randint(low, upper) for _ in range(3)])
        self.drawer.rectangle((x - r, y - 2 * r, x + r, y + 2 * r), fill=fill)

    def draw_triagnle(self, x, y, r, low, upper):
        fill = tuple([random.randint(low, upper) for _ in range(3)])
        self.drawer.polygon([(x - r, y - r), (x, y + r), (x + r, y - r)], fill=fill)

    def draw_poly(self, x, y, r, low, upper, sides):
        fill = tuple([random.randint(low, upper) for _ in range(3)])
        angle = 2 * np.pi / sides
        points = [(x + r * np.cos(i * angle), y + r * np.sin(i * angle)) for i in range(sides)]
        self.drawer.polygon(points, fill=fill)

    def draw(self, i):
        """
        Draw an image with three shapes, and draw a bounding
        box around each shape.

        Returns:
            PIL.Image.Image: The resulting image.
            list: A list of (x_min, y_min, x_max, y_max, score_1, score_2)
            coordinates for each shape, where score_1 is the shape type
            (1 = circle, 2 = triangle, 3 = rectangle, 4 = pentagon)
            and score_2 is the color (1 = dark, 2 = light).
        """
        # Initialize a new PIL image with a black background.
        img = Image.new("RGB", (self.size, self.size), (0, 0, 0))

        # Initialize a new PIL draw object to draw shapes on the image.
        self.drawer = ImageDraw.Draw(img)

        #
        boxes = []
        x, y = 0, 0
        for score_1 in range(1, 8):
            a = True
            while a:
                x = random.randint(int(0.05 * self.size), int(0.95 * self.size))
                y = random.randint(int(0.05 * self.size), int(0.95 * self.size))
                a = False
                for _ in range(x - 50, x + 50):
                    for __ in range(y - 50, y + 50):
                        try:
                            if img.getpixel((_, __))[0] != 0:
                                a = True
                                break
                        except IndexError:
                            pass
                    if a:
                        break
            r = random.randint(50, 60)
            r_ = random.randint(30, 35)

            score_2 = random.randint(1, 3)
            boxes.append([x - r_, y - r_, x + r_, y + r_, score_1, score_2 + 1])

            fill = (200, 255)
            if score_1 == 1:
                self.draw_rect(x, y, r, fill[0], fill[1])
            elif score_1 == 2:
                self.draw_circle(x, y, r, fill[0], fill[1])
            elif score_1 == 3:
                self.draw_line(x, y, r, fill[0], fill[1])
            elif score_1 == 4:
                self.draw_ellipse(x, y, r, fill[0], fill[1])
            elif score_1 == 5:
                self.draw_triagnle(x, y, r, fill[0], fill[1])
            else:
                self.draw_poly(x, y, r, fill[0], fill[1], score_1)

            if score_2 == 1:
                self.draw_rect(x, y, r_, 10, 30)
            elif score_2 == 2:
                self.draw_circle(x, y, r_, 40, 60)
            elif score_2 == 3:
                self.draw_triagnle(x, y, r_, 70, 90)

        return np.array(img.convert("L")), np.array(boxes)
