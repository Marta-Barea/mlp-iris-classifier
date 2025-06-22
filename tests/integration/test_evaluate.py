import tempfile
from pathlib import Path
from src.evaluate import evaluate_model, get_latest_model
from src.config import MODEL_OUTPUT_PATH


def test_evaluator_generates_confusion_matrix():
    model_path = get_latest_model(MODEL_OUTPUT_PATH)
    assert model_path is not None, "No .h5 model found for evaluation"

    with tempfile.TemporaryDirectory() as reports_dir:
        evaluate_model(model_path=model_path, output_dir=reports_dir)

        fig_path = Path(reports_dir) / "confusion_matrix.png"
        assert fig_path.exists(), "Confusion matrix plot was not generated"
