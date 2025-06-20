# MLP Iris Classifier

A simple project to train and evaluate a multilayer perceptron on the Iris Sepecies data using TensorFlow, SciKeras, and Scikit-Learn.

---

# Installation

1. Clone the repo

```bash
git clone https://github.com/yourusername/mlp-iris-classifier.git
cd mlp-iris-classifier
```

2. Set up the Conda environment

It is included an `environment.yml` for Conda users: 

```bash 
conda env create -f environment.yml
conda activate mlp-iris-classifier
```

# Dependencies 

- Python 3.7+
- numpy, scikt-learn, tensorflow, scikeras, PyYAML, matplotlib. 

You can also install them with:

```bash
pip install -r requirements.txt
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
- Save the best model to `models/best_mlp.h5`
- Print train/test accuracy and predictions
- Save evaluation plots to `reports/`

# Project Structure

```
mlp-iris-classifier/
│
├── config.yaml          # Experiment settings
├── environment.yml      # Conda environment spec
├── requirements.txt     # Pinned pip dependencies (for Docker)
├── docker-compose.yml   # Docker Compose setup
├── Dockerfile           # Image build definition
├── .dockerignore        # Exclude files from image context
├── .gitignore           # Exclude files from Git tracking
│
├── data/
│   └── Iris.csv         # Iris Species Dataset
│
├── models/              # (Auto-created) Trained model & params
│
├── reports/
│   └── figures          # (Auto-created) Plots
│
├── src/
│   ├── config.py        # Loads config.yaml
│   ├── data_loader.py   # Reads & splits data∫
│   ├── model_builder.py # Defines the Keras MLP
│   ├── train.py         # Hyperparameter search & model saving
│   └── evaluate.py      # Loads model & prints metrics
│
└── run_all.py           # Runs train.py then evaluate.py
```

# Dockerized Support

This project is fully containerized for portability and reproducibility.

## Docker Dependencies 

Before using Docker, you need to have the following installed locally on your system:

- [Docker Engine](https://docs.docker.com/get-started/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

✅ Note: These tools are required only if you want to run the project in a containerized environment. If you're using Conda, Docker is optional.

## How to Run 

To build the image and run the project inside a container:

```bash
docker-compose up --build
```

This will:

- Build the Docker image using the included Dockerfile
- Run the run_all.py pipeline (training + evaluation)
- Save the best trained model in the models/ directory
- Save plots and metrics in the reports/ directory

✅ Note: Both models/ and reports/ are mounted to your host machine, so your outputs are preserved outside the container.
