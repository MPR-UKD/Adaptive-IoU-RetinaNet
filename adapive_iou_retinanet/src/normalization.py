import numpy as np


class Normalizer:
    def __init__(self) -> None:
        self.mean = np.array([[0.5]])
        self.std = np.array([[1]])

    def __call__(self, sample: dict) -> dict:
        """
        Normalize the image in a sample by subtracting the mean and dividing by the standard deviation.

        Args:
            sample (dict): A dictionary containing the 'dcm' key for the image and the 'annot' key for annotations.

        Returns:
            dict: A dictionary containing the normalized image and the original annotations.
        """
        image, annots = sample["dcm"], sample["annot"]

        # Compute the mean and standard deviation of the image
        m = image.mean()
        s = image.std()

        # Normalize the image
        normalized_image = (image.astype(np.float32) - m) / s

        return {"dcm": normalized_image, "annot": annots}
