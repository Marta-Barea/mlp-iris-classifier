import os
import joblib

from .data_loader import load_data
from .config import MODEL_OUTPUT_PATH


def evaluate_model():
    X_train, X_test, y_train, y_test = load_data()
    best_model_path = os.path.join(MODEL_OUTPUT_PATH, "best_mlp.pkl")

    if not os.path.isfile(best_model_path):
        print(f"❌ Model not found at '{best_model_path}'.")
        print("   Make sure to run src/train.py first.")
        return

    best_mlp=joblib.load(best_model_path)

    train_acc=best_mlp.score(X_train, y_train)
    test_acc=best_mlp.score(X_test, y_test)

    print(f"\n📊 Train accuracy: {train_acc * 100:.2f}%")
    print(f"📊 Test accuracy:  {test_acc * 100:.2f}%")

    y_pred=best_mlp.predict(X_test)
    print("\n🔎 First 10 predictions vs. actual values:")
    for i in range(10):
        print(f"   • Predicted: {int(y_pred[i])}, Actual: {int(y_test[i])}")

if __name__ == "__main__":
    evaluate_model()
