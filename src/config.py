import os
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = load_config()

DATA_PATH = _config.get("data_path", "data/iris.csv")
MODEL_OUTPUT_PATH = _config.get("model_output_path", "model")
CHECKPOINTS_OUTPUT_PATH = _config.get("checkpoints_output_path", "checkpoints")

SEED = _config.get("seed", 42)
TEST_SIZE = _config.get("test_size", 0.3)

CV_FOLDS = _config.get("cv_folds", 3)
RANDOM_SEARCH_ITERATIONS = _config.get("random_search_iterations", 10)

search_cfg = _config.get("search", {})
UNITS_LIST = search_cfg.get("units_list", [])
EPOCHS_LIST = search_cfg.get("epochs_list", [])
BATCH_SIZE_LIST = search_cfg.get("batch_size_list", [])
LEARNING_RATE_LIST = search_cfg.get("learning_rate_list", [])
