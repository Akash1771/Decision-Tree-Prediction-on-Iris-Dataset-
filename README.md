# Decision-Tree-Prediction-on-Iris-Dataset-
# Iris Flower Prediction Model

## Overview

This project implements a predictive model that classifies iris flowers into different species based on their physical measurements. The model uses a Decision Tree Classifier from the `scikit-learn` library to make predictions. The dataset used is the well-known Iris dataset, which includes four features: sepal length, sepal width, petal length, and petal width.

## Dataset

The Iris dataset is a classic dataset in machine learning. It contains 150 samples from three different species of iris flowers:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Each sample has four features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Features

- **Data Cleaning**: The dataset is checked for missing values.
- **Feature Normalization**: Features are scaled using `StandardScaler` to improve model performance.
- **Model Training**: A Decision Tree Classifier is used to train the model on the dataset.
- **Model Evaluation**: The model is evaluated using accuracy and a confusion matrix.

## Requirements

To run this project, you need the following Python packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
