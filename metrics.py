from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np

def display_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: list
    ) -> None:
    
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\n")
    print(report)
    
def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: list
    ) -> None:
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()