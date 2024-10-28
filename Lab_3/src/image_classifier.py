import cv2
import json
import numpy as np
import sys
import time
from tqdm import tqdm  # Progress bar
from typing import List, Tuple

from bow import BoW
from dataset import Dataset
from results import Results


class ImageClassifier:
    """Class to classify images using a support vector machine (SVM) against a bag-of-words dictionary.

    Examples:
        1. Training and evaluating the classifier. Optionally, save the model.
            classifier = ImageClassifier(bow)
            classifier.train(training_set)
            classifier.predict(validation_set)
            classifier.save(filename)

        2. Loading a trained classifier to evaluate against a previously unseen test set.
            classifier = ImageClassifier(bow)
            classifier.load(filename)
            classifier.predict(test_set)

    """

    def __init__(self, bow: BoW, matcher_type: str = "FLANN"):
        """Bag-of-words initializer.

        Args:
            bow: Trained BoW object.
            matcher_type: Feature matcher { Brute-Force, FLANN }

        """
        self._labels = dict()
        self._bow = bow
        self._matcher = None
        self._classifier = None

        # Initialize dictionary from the BoW object
        self._initialize_feature_matcher(matcher_type)
        self._dictionary = cv2.BOWImgDescriptorExtractor(bow.feature_extractor, self._matcher)
        self._dictionary.setVocabulary(bow.vocabulary)

    def train(self, training_set: List[str], iterations: int = 100, epsilon: float = 1e-6):
        """Trains a SVM to classify a set of images.

        Args:
            training_set: Paths to the training images.
            iterations: Maximum number of iterations for the SVM.
            epsilon: Stop training if an accuracy of epsilon is reached.

        """
        print("\n\nTRAINING CLASSIFIER")

        # Extract BoW features from the training set
        train_desc = []
        train_labels = []
        i = 0

        print("\nExtracting features...")
        time.sleep(0.1)  # Prevents a race condition between tqdm and print statements.

        for path in tqdm(training_set, unit="image", file=sys.stdout):
            try:
                train_desc.extend(self._extract_bow_features(path))
                label = Dataset.get_label(path)

                # Convert text labels to indices starting from 0
                if label not in self._labels:
                    self._labels[label] = i
                    i += 1

                train_labels.append(self._labels[label])
            except:
                print(f"WARN: Issue Loading one label from {path}")
        # Train the classifier
        time.sleep(0.1)  # Prevents a race condition between tqdm and print statements.
        print("\nTraining SVM...")

        self._classifier = cv2.ml.SVM_create()
        self._classifier.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, iterations, epsilon))
        self._classifier.setType(cv2.ml.SVM_C_SVC)  # C-Support Vector Classification

        # Kernel type
        # - Linear: cv2.ml.SVM_LINEAR
        # - Polynomial: cv2.ml.SVM_POLY
        # - Radial basis function: cv2.ml.SVM_RBF
        # - Sigmoid: cv2.ml.SVM_SIGMOID
        # - Exponential Chi2: cv2.ml.SVM_CHI2
        # - Histogram intersection: cv2.ml.SVM_INTER
        self._classifier.setKernel(cv2.ml.SVM_LINEAR)

        # Train an SVM with optimal parameters
        self._classifier.trainAuto(
            np.array(train_desc, np.float32), cv2.ml.ROW_SAMPLE, np.array(train_labels, np.int32)
        )

    def predict(
        self, dataset: List[str], dataset_name: str = "", save: bool = True
    ) -> Tuple[float, np.ndarray, List[Tuple[str, str, str]]]:
        """Evaluates a new set of images using the trained classifier.

        Args:
            dataset: Paths to the test images.
            dataset_name: Dataset descriptive name.
            save: Save results to an Excel file.

        Returns:
            Classification accuracy.
            Confusion matrix.
            Detailed per image classification results.

        """
        # Extract features
        test_desc = []
        test_labels = []

        for path in dataset:
            descriptors = self._extract_bow_features(path)

            if descriptors is not None:
                test_desc.extend(descriptors)
                test_labels.append(self._labels.get(Dataset.get_label(path)))

        # Predict categories
        predicted_labels = (self._classifier.predict(np.array(test_desc, np.float32))[1]).ravel().tolist()
        predicted_labels = [int(label) for label in predicted_labels]

        # Format results and compute classification statistics
        results = Results(self._labels, dataset_name=dataset_name)
        accuracy, confusion_matrix, classification = results.compute(dataset, test_labels, predicted_labels)
        results.print(accuracy, confusion_matrix)

        if save:
            results.save(confusion_matrix, classification)

        return accuracy, confusion_matrix, classification

    def load(self, filename: str):
        """Loads a trained SVM model and the corresponding category labels.

        Args:
           filename: Relative path to the file up to the trailing underscore. Do not include the extension either.

        """
        # Load model
        self._classifier = cv2.ml.SVM_load(filename + "_model.xml")

        # Load labels
        with open(filename + "_labels.json") as f:
            self._labels = json.load(f)

    def save(self, filename: str):
        """Saves the model to an .xml file and the category labels to a .json file.

        Args:
           filename: Relative path to the file without the extension.

        """
        # Save model
        self._classifier.save(filename + "_model.xml")

        # Save labels
        with open(filename + "_labels.json", "w", encoding="utf-8") as f:
            json.dump(self._labels, f, ensure_ascii=False, indent=4, sort_keys=True)

    def _initialize_feature_matcher(self, matcher_type: str):
        """Initializes the feature matcher.

        Args:
            matcher_type: Feature matcher { Brute-Force, FLANN }.

        Raises:
            ValueError: If the matcher type is not known.

        """
        if matcher_type == "Brute-Force":
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif matcher_type == "FLANN":  # Fast Library for Approximate Nearest Neighbors
            index_params = dict(algorithm=0, trees=5)
            search_params = dict(checks=50)
            self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError("Matcher type not supported. Possible values are 'Brute-Force' and 'FLANN'.")

    def _extract_bow_features(self, image_path: str) -> np.ndarray:
        """Extract features using a BoW dictionary.

        Args:
            image_path: Path to the image.

        Returns:
            BoW feature (normalized histogram).

        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return self._dictionary.compute(image, self._bow.feature_extractor.detect(image))
