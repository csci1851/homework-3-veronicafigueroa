import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


def plot_classification_results(X, y, model, filename, title="SVM Decision Boundary"):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    h = 0.2
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)

    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, edgecolor="k", s=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_regression_results(y_true, y_pred, filename, title="Predicted vs Actual"):
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_residuals(y_true, y_pred, filename, title="Residual Plot"):
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def explore_classification_data(X: pd.DataFrame, y: pd.Series):
    print("Exploring classification dataset...")
    X.hist(bins=20, figsize=(15, 10))
    plt.suptitle("Feature Distributions")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Matrix")
    plt.show()

    sns.countplot(x=y)
    plt.title("Class Balance")
    plt.xlabel("Target Label")
    plt.ylabel("Count")
    plt.show()

    if X.isnull().sum().sum() > 0:
        sns.heatmap(X.isnull(), cbar=False)
        plt.title("Missing Value Map")
        plt.show()

def apply_pca(X, n_components=100):
    print(f"Running PCA on {X.shape[0]} samples, {X.shape[1]} features...")
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    print(f"Reduced to {X_reduced.shape[1]} PCA components.")
    return X_reduced