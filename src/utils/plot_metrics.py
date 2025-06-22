from pathlib import Path
import matplotlib.pyplot as plt

from ..config import REPORTS_OUTPUT_PATH


def plot_metrics(results, output_dir=None):
    output_dir = Path(output_dir or REPORTS_OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(results.history['loss'], label='train_loss')
    if 'val_loss' in results.history:
        plt.plot(results.history['val_loss'], label='val_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / 'loss.png')
    plt.close()

    plt.figure()
    plt.plot(results.history['accuracy'], label='train_accuracy')
    if 'val_accuracy' in results.history:
        plt.plot(results.history['val_accuracy'], label='val_accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(output_dir / 'accuracy.png')
    plt.close()
