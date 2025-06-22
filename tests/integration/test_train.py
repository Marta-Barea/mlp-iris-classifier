import tempfile
from pathlib import Path
from src.train import train_model


def test_trainer_creates_outputs():
    with tempfile.TemporaryDirectory() as model_dir, \
            tempfile.TemporaryDirectory() as ckpt_dir, \
            tempfile.TemporaryDirectory() as reports_dir:

        train_model(model_dir=model_dir, checkpoint_dir=ckpt_dir,
                    reports_dir=reports_dir)

        model_files = list(Path(model_dir).glob("*.h5"))
        param_files = list(Path(model_dir).glob("*.txt"))
        ckpt_files = list(Path(ckpt_dir).glob("*.h5"))
        loss_plot = Path(reports_dir) / "loss.png"
        acc_plot = Path(reports_dir) / "accuracy.png"

        assert len(model_files) >= 1, "Model file not found"
        assert len(param_files) >= 1, "Params file not found"
        assert len(ckpt_files) >= 1, "Checkpoint not found"
        assert loss_plot.exists(), "Loss plot not generated"
        assert acc_plot.exists(), "Accuracy plot not generated"
