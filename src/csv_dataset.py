from __future__ import print_function, division

import glob
import os
import random
import platform
from typing import Dict, Union, Callable, Tuple, List
from pathlib import Path
from copy import deepcopy

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as pyd
import torch
from natsort import natsorted
from torch.utils.data import Dataset
from pydicom.pixel_data_handlers.util import apply_voi_lut

from src.encoder import DataEncoder

COORDS = {
    "0": (0, 0),
    "1": (0.05747072094516853, 0.020719362394175544),
    "2": (0.08190425929348711, 0.0289776811757655),
    "3": (0.0540862908800541, 0.025809584775407236),
    "4": (0.05160626503107474, 0.027004417081198257),
    "5": (0.043246927457872834, 0.016910246847421796),
    "6": (0.054096666189375045, 0.014724664127436093),
    "7": (0.0420863770370218, 0.023763440423163563),
    "8": (0.05417839077541037, 0.03043299531369128),
    "9": (0.05493081614485686, 0.033819617426870695),
    "10": (0.053984588528493506, 0.03364476563027231),
    "11": (0.04750902450346263, 0.029393966792999392),
    "12": (0.044837214147985645, 0.028192328791580547),
    "13": (0.04332158587276059, 0.024393534674801508),
    "14": (0.04474342093399681, 0.024804890586196122),
    "15": (0.041933955856541104, 0.02957942726810968),
    "16": (0.03485362578303626, 0.03051876905506112),
    "17": (0.05768988219271516, 0.021437295781091555),
    "18": (0.11924890161031669, 0.02447149375737743),
    "19": (0.05598313910949589, 0.025076046112421948),
    "20": (0.08104923593858424, 0.027228613240540588),
    "21": (0.04209519277341975, 0.026725043810993025),
    "22": (0.05549525696573985, 0.02631286057432326),
    "23": (0.04173431176385245, 0.023142558902924687),
    "24": (0.05429669448906953, 0.029301561934338437),
    "25": (0.05589266457149611, 0.035768022040276494),
    "26": (0.05467786643347657, 0.03370090432522162),
    "27": (0.048905166101408756, 0.03047743660267201),
    "28": (0.04454138112008368, 0.030006911429529513),
    "29": (0.043425254097487834, 0.024800421234707687),
    "30": (0.0457029949136123, 0.02530248212986575),
    "31": (0.04289361006571603, 0.024211364415482058),
    "32": (0.03626870039542136, 0.021178056613016555),
}


class CSVDataset(Dataset):
    """
    Custom dataset class to load data from a CSV file.
    """

    def __init__(
        self,
        data_folder: str,
        csv_file: str,
        config: dict,
        mode: str = "train",
        distribution: list = [0.7, 0.1],
        transformer: object = Union[None, Callable],
        pos: float = 0.5,
        neg: float = 0.3,
        adaptive_epochs: int = 0,
        epoch_dict: dict = None,
    ):
        """
        Initialize the CSVDataset object.

        The CSVDataset class is a custom PyTorch dataset class used to load data from a CSV file. The class can be used
        to load data for training, validation, or testing. The CSV file must contain annotations for each data sample.

        Args:
            data_folder (str): The path to the folder containing the data samples.
            csv_file (str): The path to the CSV file containing the data annotations.
            config (dict): A dictionary containing configuration parameters for the model.
            mode (str, optional): The dataset mode, one of 'train', 'val', or 'test'. Defaults to 'train'.
            distribution (list, optional): The data distribution for training and validation. Defaults to [0.7, 0.1].
            transformer (object, optional): A PyTorch transformation object to apply to the data. Defaults to None.
            pos (float, optional): The positive sample weight for the loss function. Defaults to 0.5.
            neg (float, optional): The negative sample weight for the loss function. Defaults to 0.3.
            adaptive_epochs (int, optional): The number of epochs for adaptive sampling. Defaults to 0.
            epoch_dict (dict, optional): A dictionary containing information about the current epoch. Defaults to None.

        Returns:
            None
        """
        self.test_mode = True if platform.system() == "Windows" else False
        self.transform = transformer
        self.distribution = distribution
        self.data, cls_num_1, cls_num_2 = self.load_data(data_folder, csv_file, mode)
        self.encoder = DataEncoder(cf=config)
        self.num_cls = [cls_num_1 + 1, cls_num_2 + 1]
        self.mode = mode

        self.epoch = epoch_dict
        self.pos = pos
        self.neg = neg
        self.adaptive_epochs = adaptive_epochs

    def load_data(self, data_folder, csv_file, mode):
        """
        Load and preprocess data from a given folder and CSV file according to the specified mode.

        Args:
            data_folder (str): Path to the folder containing DICOM images.
            csv_file (str): Path to the CSV file containing labels for each image.
            mode (str): One of {'train', 'val', 'test'}, indicating the data split to load.

        Returns:
            tuple: A tuple containing the loaded data, the number of samples in class 1, and the number of samples in class 2.
        """
        data, cls_num_1, cls_num_2 = load_data(data_folder, csv_file)

        # Shuffle the data randomly
        random.seed(1)
        random.shuffle(data)

        train = round(self.distribution[0] * len(data))
        val = round(self.distribution[1] * len(data))
        repetitions = 4

        if mode == "train":
            if self.test_mode:
                # If test mode is enabled, return a small subset of data for quick testing
                return data[:2], cls_num_1, cls_num_2

            # Return training data and repeat each sample multiple times to balance the classes
            return data[:train] * repetitions, cls_num_1, cls_num_2

        elif mode == "val":
            if self.test_mode:
                # If test mode is enabled, return a small subset of data for quick testing
                return data[:2], cls_num_1, cls_num_2

            # Return validation data
            return data[train : train + val], cls_num_1, cls_num_2

        elif mode == "test":
            # Return test data
            return data[train + val :], cls_num_1, cls_num_2

        else:
            # If an invalid mode is specified, return all the data
            return data, cls_num_1, cls_num_2

    def test_run(
        self, data: List[Dict], train_size: int, val_size: int
    ) -> Tuple[List[Dict], int, int]:
        """
        Split the input data into train and validation sets, and return the first two samples of the train set.

        Args:
            data (List[Dict]): A list of dictionaries containing information about each data sample.
            train_size (int): The size of the train set.
            val_size (int): The size of the validation set.

        Returns:
            Tuple[List[Dict], int, int]: A tuple containing the first two samples of the train set, and the sizes of the two sets.
        """
        train_set = data[:train_size]
        val_set = data[train_size : train_size + val_size]
        return train_set[:2], len(train_set), len(val_set)

    def num_classes(self) -> List:
        """
        Get the number of unique classes in the input data.

        Returns:
            int: The number of unique classes.
        """
        return self.num_cls

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
        # Compute the positive and negative sample ratios based on the current epoch and adaptive epochs
        pos_ratio = (
            self.pos / self.adaptive_epochs * self.epoch["epoch"]
            if self.adaptive_epochs != 0
            else self.pos
        )
        neg_ratio = (
            self.neg / self.adaptive_epochs * self.epoch["epoch"]
            if self.adaptive_epochs != 0
            else self.neg
        )

        # Limit the positive and negative sample ratios to the maximum values
        pos_ratio = pos_ratio if pos_ratio < self.pos else self.pos
        neg_ratio = neg_ratio if neg_ratio < self.neg else self.neg

        # Set a minimum positive sample ratio to avoid division by zero
        pos_ratio = pos_ratio if pos_ratio > 0 else 0.05

        # Ensure that the positive sample ratio is greater than the negative sample ratio
        assert pos_ratio > neg_ratio

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
        data = self.data[idx]
        dcm_file = data["dcm_file"]
        dcm = np.array(load_dicom(dcm_file))
        annotations = self.load_annotations(idx)

        # Prepare the item data as a dictionary
        count = data["count"]
        data_dict = {
            "dcm": dcm.astype("float32"),
            "annot": annotations.astype("float32"),
        }

        # Apply the transformation function to the item data, if provided
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        # Add metadata to the item data dictionary
        data_dict["dcm_file"] = dcm_file
        data_dict["annot_org"] = data_dict["annot"]
        data_dict["annot"] = self.transform_annotations(
            torch.from_numpy(data_dict["annot"]), data_dict["dcm"].shape
        )

        # Add a batch size dimension to the item data tensors
        data_dict["annot_org"] = torch.tensor(
            np.expand_dims(data_dict["annot_org"], axis=0)
        )
        data_dict["annot"] = torch.tensor(np.expand_dims(data_dict["annot"], axis=0))
        data_dict["dcm"] = torch.tensor(
            np.expand_dims(data_dict["dcm"].astype("float32"), axis=0)
        )
        return data_dict


def show_data(data: Dict[str, Union[str, np.ndarray]]) -> None:
    """
    Display the input data in a Matplotlib plot.

    Args:
        data (Dict[str, Union[str, np.ndarray]]): A dictionary containing the input data and metadata.
    """
    # Load the DICOM image from the input data dictionary
    dcm_file = data["dcm_file"]
    img = load_dicom(dcm_file)

    # Create figure and axes for the plot
    fig, ax = plt.subplots()

    # Display the DICOM image
    ax.imshow(img)

    # Add rectangles to the plot for each set of coordinates in the input data
    for coords in data["coords"]:
        # Create a rectangle patch
        rect = patches.Rectangle(
            (int(coords[0]) - 20, int(coords[1]) - 20),
            40,
            40,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

    # Show the plot
    plt.show()


def load_dicom_folder(dcm_folder: str) -> Dict[str, str]:
    """
    Load DICOM files from a folder and store them in a dictionary.

    Args:
        dcm_folder (str): The path to the folder containing the DICOM files.

    Returns:
        Dict[str, str]: A dictionary of DICOM files keyed by patient number.
    """
    # Initialize an empty dictionary to store the DICOM files
    patients = {}

    # Load DICOM files from the folder and store them in the dictionary keyed by patient number
    for file in natsorted(glob.glob(dcm_folder + os.sep + "*")):
        pat_nr = os.path.basename(file)
        pat_nr = pat_nr.split("_")[-1]
        patients[pat_nr] = file

    return patients


def find_measurement_time_point(pat_file: str, nr: int) -> Union[str, None]:
    """
    Find the DICOM file for a specific measurement time point.

    Args:
        pat_file (str): The path to the folder containing the DICOM files for a patient.
        nr (int): The measurement time point to find.

    Returns:
        Union[str, None]: The path to the DICOM file for the specified measurement time point, or None if not found.
    """
    try:
        # Find the DICOM file for the specified measurement time point
        return natsorted(glob.glob(pat_file + os.sep + "*"))[nr - 1]
    except IndexError:
        # Return None if the specified measurement time point is not found
        return None


def find_dicom(path_measurement: str, name: str = "Handvd") -> Union[str, None]:
    """
    Find a DICOM file in a directory hierarchy with a specific name.

    Args:
        path_measurement (str): The path to the directory hierarchy.
        name (str, optional): The name of the DICOM file to find. Defaults to 'Handvd'.

    Returns:
        Union[str, None]: The path to the DICOM file if found, or None if not found.
    """
    # Get a sorted list of all files in the directory hierarchy
    files = natsorted(glob.glob(path_measurement + os.sep + "*"))

    # Find all DICOM files with the specified name and return the first one
    dcm_files = []
    for file in files:
        if name in file:
            dcm_files.append(natsorted(glob.glob(file + os.sep + "*"))[0])
    if len(dcm_files) != 1:
        return None
    else:
        return dcm_files[0]


def load_dicom(dcm_file: Path):
    mode = 2
    if mode == 1:
        ds = pyd.dcmread(dcm_file)
        arr = ds.pixel_array
        out = apply_voi_lut(arr, ds)
        return out
    elif mode == 2:
        ds = pyd.dcmread(dcm_file)
        window_center = float(ds.WindowCenter)
        window_width = float(ds.WindowWidth)
        dicom = ds.pixel_array

        # View dicom images with correct windowing
        dicom_min = window_center - window_width // 2
        dicom_max = window_center + window_width // 2
        dicom = dicom - dicom_min
        dicom[dicom < 0] = 0
        dicom[dicom > dicom_max] = dicom_max
        return 4095 * dicom / dicom.max()
    else:
        return pyd.dcmread(dcm_file).pixel_array


def load_data(
    dcm_folder: str, csv_file: str
) -> Tuple[List[Dict[str, Union[List[Tuple[str]], List[List[int]]]]], int, int]:
    """Load data from DICOM and CSV files.

    Args:
        dcm_folder: The folder path containing DICOM files.
        csv_file: The CSV file path containing labels.

    Returns:
        A tuple of:
            - A list of dictionaries containing coordinates, bounding boxes, scores, DICOM file path, and count.
            - The maximum score_1 value from the CSV file.
            - The maximum score_2 value from the CSV file.
    """
    # Load DICOM files and CSV file
    patients = load_dicom_folder(dcm_folder)
    df = pd.read_excel(csv_file, sheet_name="Erosion")

    # Initialize variables
    data, current_pat, na, max_score_1, max_score_2, min_score_2, min_score_1 = (
        [],
        "",
        0,
        -1,
        -1,
        10**3,
        10**3,
    )

    # Iterate through each row in the CSV file
    for i in range(df.shape[0]):
        measurement_nr = str(df.iloc[i, 0]).split("_")
        if measurement_nr[0] != current_pat:
            current_pat = measurement_nr[0]
            na = 0
        if not df.iloc[i, 1] in ["X", "O"]:
            na += 1
        elif df.iloc[i, 1] in ["X"]:

            try:
                measurement_file = find_measurement_time_point(
                    patients[measurement_nr[0]],
                    int(measurement_nr[-1]) - na if len(measurement_nr) > 1 else 1,
                )
            except KeyError:
                continue
            if measurement_file is None:
                continue
            dcm_file = find_dicom(measurement_file)

            if dcm_file is None:
                continue
            img_size = load_dicom(dcm_file).shape

            # Get the coordinates and scores from the row in the CSV file
            coords, scores = get_coord_score(df.iloc[i])
            bboxes = []

            # Iterate through each coordinate and score pair and calculate the bounding box
            for j, (coord, score) in enumerate(zip(coords, scores)):
                score_1, score_2 = score
                max_score_1 = score_1 if score_1 > max_score_1 else max_score_1
                max_score_2 = score_2 if score_2 > max_score_2 else max_score_2

                x_size = img_size[0] * COORDS[str(score_1)][0]
                y_size = img_size[1] * COORDS[str(score_1)][1]

                bboxes.append(
                    [
                        int(coord[0]) - x_size,
                        int(coord[1]) - y_size,
                        int(coord[0]) + x_size,
                        int(coord[1]) + y_size,
                    ]
                )

            # Add the data to the list of dictionaries
            data.append(
                {
                    "coords": coords,
                    "bboxes": bboxes,
                    "scores": scores,
                    "dcm_file": dcm_file,
                    "count": i,
                }
            )

    # Return the list of dictionaries, maximum score_1, and maximum score_2
    return data, max_score_1, max_score_2


def get_coord_score(row):
    """
    Extracts coordinates and scores from a row in the CSV file.

    Args:
        row (pd.Series): A row from the CSV file containing coordinate and score information.

    Returns:
        tuple: A tuple containing two lists: one list of coordinate tuples (x, y), and one list of score tuples (score_1, score_2).
    """
    coords = []
    scores = []
    count = 0
    for x, y in zip(row[2::2], row[3::2]):
        count += 1
        if x == "O":
            continue
        try:
            coord = tuple(x.split("/"))
        except:
            coord = (0, 0, 0)
        if len(coord) != 2:
            coord, score_1, score_2 = (0, 0), 0, 0
        else:
            try:
                score_1, score_2 = count, int(y) + 1
            except:
                coord, score_1, score_2 = (0, 0), 0, 0
        coords.append(coord)
        scores.append((score_1, score_2))
    return coords, scores


def show_images(data: dict, number: int) -> None:
    """
    Display an image with annotations and save it to a file.

    Args:
        data (dict): The data to display.
        number (int): The number to use in the output filename.
    """
    img, annont_org = data["dcm"].cpu().numpy(), data["annot_org"].cpu().numpy()
    for b in range(img.shape[0]):
        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(img[b], cmap="gray")

        for i, coord in enumerate(annont_org[b]):
            label = str(int(coord[4].item())) + "_" + str(int(coord[5].item()))
            rect = patches.Rectangle(
                (coord[0], coord[1]),
                coord[2] - coord[0],
                coord[3] - coord[1],
                linewidth=1,
                edgecolor="b",
                facecolor="none",
            )

            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.text(coord[0], coord[1], label, color="white", fontsize="xx-small")

        plt.axis("off")
        plt.savefig(f".{os.sep}images{os.sep}{number}.png", bbox_inches="tight")
        plt.close()
