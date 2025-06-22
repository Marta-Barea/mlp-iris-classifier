import pytest
from src import data_loader


def test_load_data_returns_four_arrays():
    len_arrays = len(data_loader.load_data())
    assert len_arrays == 4


def test_load_data_shapes():
    X_train, X_test, y_train, y_test = data_loader.load_data()

    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_load_data_not_empty():
    X_train, X_test, y_train, y_test = data_loader.load_data()

    assert X_train.size > 0
    assert X_test.size > 0
    assert y_train.size > 0
    assert y_test.size > 0


def test_load_data_file_not_found():
    with pytest.raises(OSError):
        data_loader.load_data(path="nonexistent_file.csv")
