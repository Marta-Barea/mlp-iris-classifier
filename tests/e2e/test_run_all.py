import tempfile
from pathlib import Path
from src.train import train_model
from src.evaluate import evaluate_model, get_latest_model


def test_full_pipeline_e2e():
    with tempfile.TemporaryDirectory() as model_dir, \
            tempfile.TemporaryDirectory() as ckpt_dir, \
            tempfile.TemporaryDirectory() as report_dir:

        train_model(model_dir=model_dir, checkpoint_dir=ckpt_dir,
                    reports_dir=report_dir)

        model_path = get_latest_model(model_dir)
        assert model_path is not None and Path(
            model_path).exists(), "Model not saved"
        param_files = list(Path(model_dir).glob("*.txt"))
        assert param_files, "Param file not saved"

        evaluate_model(model_path=model_path, output_dir=report_dir)

        confusion_path = Path(report_dir) / "confusion_matrix.png"
        loss_path = Path(report_dir) / "loss.png"
        acc_path = Path(report_dir) / "accuracy.png"

        assert confusion_path.exists(), "Confusion matrix not generated"
        assert loss_path.exists(), "Loss plot not generated"
        assert acc_path.exists(), "Accuracy plot not generated"
