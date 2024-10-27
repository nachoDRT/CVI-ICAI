import cv2
import numpy as np
import pickle
import sys
import time
from tqdm import tqdm  # Progress bar
from typing import List


class BoW:
    """Class to build a bag-of-words (bag-of-features) for image classification.

    Examples:
        1. Building a new vocabulary. Optionally save it for future use.
            bow = BoW()
            bow.build_vocabulary(training_set, vocabulary_size=500)
            bow.save_vocabulary(filename)

        2. Loading a previously built vocabulary
            bow = BoW()
            bow.load_vocabulary(filename)

    """

    def __init__(self):
        """Bag-of-words initializer."""
        self._feature_type = ""
        self._feature_extractor = None
        self._vocabulary = []

    @property
    def feature_extractor(self):
        """Return the feature extractor object."""
        return self._feature_extractor

    @property
    def vocabulary(self):
        """Return the vocabulary."""
        return self._vocabulary

    def build_vocabulary(
        self,
        training_set: List[str],
        feature_type: str = "SIFT",
        vocabulary_size: int = 100,
        iterations: int = 100,
        epsilon: float = 1e-6,
    ):
        """Builds a dictionary by clustering all the descriptors in the training set using K-means.

        Args:
            training_set: Paths to the training images.
            feature_type: Feature extractor { SIFT, KAZE }.
            vocabulary_size: Number of clusters.
            iterations: Maximum number of iterations for K-means.
            epsilon: Stop K-means if an accuracy of epsilon is reached.

        """
        print("\nBUILDING DICTIONARY")

        self._initialize_feature_extractor(feature_type)
        termination_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, iterations, epsilon)
        words = cv2.BOWKMeansTrainer(vocabulary_size, termination_criteria)

        # Extract features
        print("\nComputing", feature_type, "descriptors...")
        time.sleep(0.1)  # Prevents a race condition between tqdm and print statements.

        for path in tqdm(training_set, unit="image", file=sys.stdout):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            try:
                _, descriptor = self._feature_extractor.detectAndCompute(image, None)
            except:
                print(f"WARN: Issue generating descriptor for image {path}")

            if descriptor is not None:
                words.add(descriptor)

        # Build vocabulary
        time.sleep(0.1)  # Prevents a race condition between tqdm and print statements.
        print("\nClustering descriptors into", vocabulary_size, "words using K-means...")

        self._vocabulary = words.cluster()

    def load_vocabulary(self, filename: str):
        """Loads a pre-trained vocabulary from a .pickle file.

        Args:
            filename: Relative path to the file without the extension.

        """
        with open(filename + ".pickle", "rb") as f:
            feature_type, self._vocabulary = pickle.load(f)

        self._initialize_feature_extractor(feature_type)

    def save_vocabulary(self, filename: str):
        """Saves the vocabulary to a .pickle file to prevent having to build it every time.

        Args:
           filename: Relative path to the file without the extension.

        """
        with open(filename + ".pickle", "wb") as f:
            pickle.dump([self._feature_type, self._vocabulary], f, pickle.HIGHEST_PROTOCOL)

    def _initialize_feature_extractor(self, feature_type: str):
        """Initializes the feature extractor.

        Args:
            feature_type: Feature extractor { SIFT, KAZE }.

        Raises:
            ValueError: If the feature type is not known.

        """
        if feature_type == "SIFT":
            self._feature_extractor = cv2.SIFT_create()
        elif feature_type == "KAZE":
            self._feature_extractor = cv2.KAZE_create()
        else:
            raise ValueError("Feature type not supported. Possible values are 'SIFT' and 'KAZE'.")

        self._feature_type = feature_type
