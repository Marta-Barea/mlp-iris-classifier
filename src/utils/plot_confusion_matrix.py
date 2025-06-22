import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from ..config import REPORTS_OUTPUT_PATH


def plot_confusion_matrix(model, X, y, output_dir=None):
    output_dir = Path(output_dir or REPORTS_OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = (y_pred > 0.5).astype(int).ravel()

    y_true = y.ravel() if y.ndim > 1 else y
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    labels = np.unique(y_true)
    class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    output_path = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(output_path)
    fig.canvas.draw()
    plt.close(fig)
