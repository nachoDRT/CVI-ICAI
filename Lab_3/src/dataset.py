import glob
import os
from sklearn.model_selection import train_test_split
from typing import List, Tuple


class Dataset:
    """Class to handle image datasets organized in folders according to their category.

    Examples:
        1. Load a dataset and split it into training (70%) and validation (30%) sets.
            training_set = Dataset.load('../dataset/training', '*.jpg')
            training_set, validation_set = Dataset.split(training_set, 0.7)

    """

    @staticmethod
    def load(directory: str, file_extension: str) -> List[str]:
        """Reads the paths of a set of images organized in folders according to their category.

        Args:
            directory: Relative path to the root folder (e.g., '../dataset').
            file_extension: File extension (e.g., '*.jpg').

        Returns:
            List of full p aths to every file with the specified extension (e.g., '../dataset/label/image.jpg').

        """
        return sorted(glob.glob(directory + '/**/*' + file_extension, recursive=True))

    @staticmethod
    def get_label(path: str) -> str:
        """Returns the category of a given image described by its path.

        Args:
            path: Full path to an image, including the filename and the extension (e.g., '../dataset/label/image.jpg').

        Returns:
            Image category (e.g. label).

        """
        return os.path.basename(os.path.dirname(path))

    @staticmethod
    def split(dataset: List[str], training_size: float) -> Tuple[List[str], List[str]]:
        """Splits a dataset into training and validation (or test) randomly.

        Args:
            dataset: Paths to the images.
            training_size: Size of the resulting training set [0.0, 1.0].

        Raises:
            ValueError: If training_size is out of range.

        Returns:
            Training set.
            Validation (or test) set.

        """
        if training_size < 0.0 or training_size > 1.0:
            raise ValueError("training_size must be a number in the range [0.0, 1.0].")

        test_size = 1.0 - training_size

        return train_test_split(dataset, test_size=test_size, shuffle=True)
