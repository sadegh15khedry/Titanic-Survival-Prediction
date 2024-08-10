## Overview
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

### Training

- Train Loss: 0.3781 
- Train Accuracy: 0.8296 

### Validation
- Val Loss: 0.5708 
- Val Accuracy: 0.8146
  
![training_validation_loss_and_accuracy](https://github.com/user-attachments/assets/4d3f5751-f80a-4f06-874e-1636334e9d5f)
** Train and validation loss and accuracy **

### Testing

- Test accuracy: 0.754
- Test Loss: 0.669
- Precision: 0.753
- recall: 0.754
- F1: 0.749

```
              precision    recall  f1-score   support

           0       0.76      0.86      0.80       105
           1       0.75      0.61      0.67        74

    accuracy                           0.75       179
   macro avg       0.75      0.73      0.74       179
weighted avg       0.75      0.75      0.75       179
```

![cnfusion_matrix](https://github.com/user-attachments/assets/ce784542-431f-455d-bdaa-b3d529c31a8c)
** Confusion Matrix **

## Future Work

- **Hyperparameter Tuning:** Experiment with different architectures, learning rates, and other hyperparameters.
- **Ensemble Methods:** Combine predictions from multiple models to improve accuracy.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
