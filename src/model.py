from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def svm_classifier(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale"):
    """
    TODO: Return a scikit-learn SVC model with the specified parameters.
    """
    svc = SVC(kernel = kernel, C = C, degree = degree, gamma = gamma)
    return svc


def svm_regressor(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale"):
    """
    TODO: Return a scikit-learn SVR model with the specified parameters.
    """
    svr = SVR(kernel = kernel, C = C, degree = degree, gamma = gamma)
    return svr

def evaluate_classifier(model, X_test, y_test):
    """
    TODO: Compute and return accuracy, precision, recall, and F1 score
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return metrics



def evaluate_regressor(model, X_test, y_test):
    """
    TODO: Compute and return MAE, RMSE, and R2
    """
    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    return metrics