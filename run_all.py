import os
import sys
import importlib
import pathlib

from src.train import train_model
from src.evaluate import evaluate_model


def runner():
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(project_root))

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    print("=== STARTING TRAINING ===")
    try:
        train_model()
    except Exception as e:
        print("❌ Error during training:")
        print(e)
        sys.exit(1)

    print("\n=== STARTING EVALUATION ===")
    try:
        evaluate_model()
    except Exception as e:
        print("❌ Error during evaluation:")
        print(e)
        sys.exit(1)

    print("\n=== PROCESS COMPLETED ===")


if __name__ == "__main__":
    runner()
