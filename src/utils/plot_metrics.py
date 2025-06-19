import os
import matplotlib.pyplot as plt


def plot_metrics(results, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(results.history['loss'], label='train_loss')
    if 'val_loss' in results.history:
        plt.plot(results.history['val_loss'], label='val_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(results.history['accuracy'], label='train_accuracy')
    if 'val_accuracy' in results.history:
        plt.plot(results.history['val_accuracy'], label='val_accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()
