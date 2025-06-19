import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(model, X, y, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    y_true = y.ravel() if y.ndim > 1 else y

    labels = [0, 1]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig)
