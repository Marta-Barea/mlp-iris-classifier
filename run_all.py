import sys
import pathlib
from src.train import train_model
from src.evaluate import evaluate_model, get_latest_model


def runner(model_dir=None, checkpoint_dir=None, reports_dir=None):
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(project_root))

    print("=== STARTING TRAINING ===")
    try:
        train_model(model_dir=model_dir, checkpoint_dir=checkpoint_dir)
    except Exception as e:
        print("❌ Error during training:")
        print(e)
        sys.exit(1)

    print("\n=== STARTING EVALUATION ===")
    try:
        evaluate_model(output_dir=reports_dir)
    except Exception as e:
        print("❌ Error during evaluation:")
        print(e)
        sys.exit(1)

    print("\n=== PROCESS COMPLETED ===")


if __name__ == "__main__":
    runner()
