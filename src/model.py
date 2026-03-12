from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def svm_classifier(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale"):
    """
    TODO: Return a scikit-learn SVC model with the specified parameters.
    """
    pass


def svm_regressor(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale"):
    """
    TODO: Return a scikit-learn SVR model with the specified parameters.
    """
    pass

def evaluate_classifier(model, X_test, y_test):
    """
    TODO: Compute and return accuracy, precision, recall, and F1 score
    """
    pass


def evaluate_regressor(model, X_test, y_test):
    """
    TODO: Compute and return MAE, RMSE, and R2
    """
    pass