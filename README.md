# Titanic Survival Prediction

This repository contains a machine learning project for predicting survival on the Titanic using a neural network.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict whether a passenger survived the Titanic disaster based on various features such as age, gender, class, etc. We employ various data preprocessing techniques and build an Artificial Neural Network (ANN) to make these predictions.

## Dataset

The dataset used in this project is the Titanic dataset available from [Kaggle](https://www.kaggle.com/c/titanic/data). It contains information about the passengers including whether they survived or not.

## Project Structure

The project has the following structure:
```
titanic-survival-prediction/
│
├── datasets/
│ ├── raw/
│ │ └── titanic.csv
│ └── read/
│ ├── train/
│ │ ├── X_train.csv
│ │ └── y_train.csv
│ ├── val/
│ │ ├── X_val.csv
│ │ └── y_val.csv
│ └── test/
│ ├── X_test.csv
│ └── y_test.csv
│
├── notebooks/
│ ├── data_exploration.ipynb
│ ├── data_preprocessing.ipynb
│ ├── model_training.ipynb
│ └── model_evaluation.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ └── utils.py
│
├── results/
│ ├── model_accuracy.png
│ ├── model_loss.png
│
├── README.md
└── environment.yml
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sadegh15khedry/Titanic-Survival-Prediction
   cd Titanic-Survival-Prediction
   ```
2. **Create and activate a conda environment:**
  ```bash
   conda env create -f environment.yml
   conda activate titanic
   ```
3. **Download the dataset:**

   - Download the dataset from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/data).
   - Place the downloaded files in the `datasets/raw/` directory.
   - Run the preprocessing script to split the data into train, validation, and test sets and place them in the respective folders in `datasets/ready/`.

## Data Preprocessing

Data preprocessing involves handling missing values, encoding categorical variables, and normalizing numerical features. The preprocessing steps can be found in the `data_preprocessing.py` script and are also detailed in the `data_preprocessing.ipynb` notebook.

## Model Training

We use an Artificial Neural Network (ANN) to predict survival. The network architecture, training procedure, and evaluation are implemented in `model_training.py` and `model_training.ipynb`.


## Model Evaluation

The model's performance is evaluated on the validation set. Metrics such as accuracy, precision, recall, and F1-score are used to assess the model. Evaluation results and visualizations can be found in `model_evaluation.py` and `model_evaluation.ipynb`.


## Results

The model achieved an accuracy of 75% on the training set and 72% on the validation set. Further improvements and experimentation are needed to enhance the performance.

## Future Work

- **Hyperparameter Tuning:** Experiment with different architectures, learning rates, and other hyperparameters.
- **Feature Engineering:** Create new features based on domain knowledge.
- **Ensemble Methods:** Combine predictions from multiple models to improve accuracy.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
