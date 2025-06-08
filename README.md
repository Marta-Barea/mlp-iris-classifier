# MLP Iris Classifier

A simple project to train and evaluate a multilayer perceptron on the Iris Sepecies data using TensorFlow, SciKeras, and Scikit-Learn.

---

# Installation

1. Clone the repo

```bash
git clone https://github.com/yourusername/mlp-iris-classifier.git
cd mlp-iris-classifier
```

2. Create a Conda enviornment

It is included an `environment.yml` for Conda users: 

```bash 
conda env create -f environment.yml
conda activate mlp-iris-classifier
```

# Usage

1. Verify the dataset

The [Iris Species Dataset](https://archive.ics.uci.edu/dataset/53/iris) from the UCI Machine Learning Repository is already included under data/Iris.csv.

2. Adjust settings

Open `config.yaml`and tweak any values you like (seed, test_size_hyperparameters list, etc.)

3. Run the full pipeline

```bash
python run_all.py
```

This will: 

- Train de MLP with randomized hyperparameter search
- Save the best model to `models/best_mlp.pk`
- Evaluate and print train/test accuracy and sample predictions

# Project Structure

mlp-iris-classifier/
│
├── config.yaml          # Experiment settings
├── environment.yml      # Conda environment spec
│
├── data/
│   └── Iris.csv     # Iris Species Dataset
│
├── models/              # (Auto-created) Trained model & params
│
├── src/
│   ├── config.py        # Loads config.yaml
│   ├── data_loader.py   # Reads & splits data
│   ├── model_builder.py # Defines the Keras MLP
│   ├── train.py         # Hyperparameter search & model saving
│   ├── evaluate.py      # Loads model & prints metrics
│   └── utils.py         # (Optional) Helper functions
│
└── run_all.py           # Runs train.py then evaluate.py

# Dependencies 

- Python 3.7+
- numpy, scikt-learn, tensorflow, scikeras, joblib, PyYAML

With Conda:

```bash 
conda env create -f environment.yml
conda activate mlp-iris-classifier
```