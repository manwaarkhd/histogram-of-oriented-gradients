from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from skimage.feature import hog

import numpy as np
import cv2

class HistogramOfOrientedGradients:
    
    def __init__(
        self,
        classifier: [LinearSVC, RandomForestClassifier]
    ) -> None:
        
        self.classifier = classifier()

    def train(
        self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> None:

        feature_descriptors = []

        for image in images:

            descriptor = hog(
                image,
                orientations = 9, 
                pixels_per_cell = (8, 8), 
                cells_per_block = (2, 2), 
                visualize = False
            )

            feature_descriptors.append(descriptor)

        X = np.array(feature_descriptors)
        y = labels.ravel()

        self.classifier.fit(X, y)

    def test(
        self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> None:
        
        num_instances = images.shape[0]
        feature_descriptors = []

        for image in images:

            descriptor = hog(
                image,
                orientations = 9, 
                pixels_per_cell = (8, 8), 
                cells_per_block = (2, 2), 
                visualize = False
            )

            feature_descriptors.append(descriptor)

        X = np.array(feature_descriptors)
        y_true = labels.ravel()

        y_pred = self.classifier.predict(X)
        
        correct = (y_true == y_pred).sum()
        accuracy = accuracy_score(y_true, y_pred) * 100
        
        print(f"        Total instances: {num_instances}")
        print(f"Correct classifications: {correct}")
        print(f"         Accuracy score: {round(accuracy, 3)}")

        return y_pred