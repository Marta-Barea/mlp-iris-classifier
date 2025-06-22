import src.config as config


def test_config_constants_exist():
    assert isinstance(config.SEED, int)
    assert isinstance(config.TEST_SIZE, float)
    assert isinstance(config.CV_FOLDS, int)
    assert isinstance(config.RANDOM_SEARCH_ITERATIONS, int)


def test_config_paths_are_strings():
    assert isinstance(config.DATA_PATH, str)
    assert isinstance(config.MODEL_OUTPUT_PATH, str)
    assert isinstance(config.CHECKPOINTS_OUTPUT_PATH, str)
    assert isinstance(config.REPORTS_OUTPUT_PATH, str)


def test_config_lists_are_loaded():
    assert isinstance(config.UNITS_LIST, list)
    assert isinstance(config.LEARNING_RATE_LIST, list)
    assert isinstance(config.EPOCHS_LIST, list)
    assert isinstance(config.BATCH_SIZE_LIST, list)
