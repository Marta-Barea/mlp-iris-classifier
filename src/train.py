import os
import random
import numpy as np
import tensorflow as tf

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib

from src.data_loader import load_data
from src.build_model import build_model

from .config import (
    SEED,
    UNITS_LIST,
    EPOCHS_LIST,
    BATCH_SIZE_LIST,
    LEARNING_RATE_LIST,
    RANDOM_SEARCH_ITERATIONS,
    MODEL_OUTPUT_PATH,
    CV_FOLDS
)


def train_model():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, X_test, y_train, y_test = load_data()

    input_dim = X_train.shape[1]

    default_units = UNITS_LIST[0] if UNITS_LIST else 8
    default_lr = LEARNING_RATE_LIST[0] if LEARNING_RATE_LIST else 0.001

    mlp = KerasClassifier(
        model=build_model,
        input_dim=X_train.shape[1],
        units=default_units,
        learning_rate=default_lr,
    )

    param_dist = {
        "model__units": UNITS_LIST,
        "epochs": EPOCHS_LIST,
        "batch_size": BATCH_SIZE_LIST,
        "model__learning_rate": LEARNING_RATE_LIST
    }

    search = RandomizedSearchCV(
        mlp,
        param_distributions=param_dist,
        n_iter=RANDOM_SEARCH_ITERATIONS,
        cv=CV_FOLDS,
        verbose=2,
        random_state=SEED,
    )

    search.fit(X_train, y_train)

    print("⏳ Starting RandomizedSearchCV...")
    search.fit(X_train, y_train)

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

    best_model_path = os.path.join(MODEL_OUTPUT_PATH, "best_mlp.pkl")
    joblib.dump(search.best_estimator_, best_model_path)

    best_params_path = os.path.join(MODEL_OUTPUT_PATH, "best_params.txt")
    with open(best_params_path, "w") as f:
        f.write(str(search.best_params_))

    print("\n✅ Training completed.")
    print(f"   • Best model saved at: {best_model_path}")
    print(f"   • Best parameters saved at: {best_params_path}")
    print("\n🔍 Best parameters found:")
    for key, value in search.best_params_.items():
        print(f"     • {key}: {value}")


if __name__ == "__main__":
    train_model()
