import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import DATA_PATH, SEED, TEST_SIZE


def load_data(path=DATA_PATH):
    iris_data = pd.read_csv(path, skiprows=1, header=None)
    iris_data = iris_data.drop(columns=[0])
    arr = iris_data.values

    X_iris = arr[:, :-1].astype(float)
    y_text = arr[:, -1]

    encoder = LabelEncoder()
    y_iris = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=TEST_SIZE, random_state=SEED
    )

    return X_train, X_test, y_train, y_test
