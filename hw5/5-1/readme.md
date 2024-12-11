# Iris Classification with PyTorch Lightning

This repository provides a PyTorch Lightning implementation for classifying iris flowers into three species based on their features. The project follows the CRISP-DM methodology and leverages PyTorch Lightning for model training, validation, and testing.

## Project Workflow (CRISP-DM Steps)

### 1. Business Understanding
The goal is to classify iris flowers into one of three species (`setosa`, `versicolor`, `virginica`) based on four features: sepal length, sepal width, petal length, and petal width.

### 2. Data Understanding
- Dataset: Iris dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- The dataset contains 150 samples evenly distributed across three species.

### 3. Data Preparation
- Features are standardized using `StandardScaler` from `sklearn`.
- Labels are encoded to integers for classification.
- Data is split into training (80%) and testing (20%) sets.
- PyTorch `TensorDataset` and `DataLoader` are used for batching and shuffling.

### 4. Modeling
- The neural network has three layers:
  - Input to hidden layer with 64 neurons and a `ReLU` activation function.
  - Hidden layer to another layer with 32 neurons and `ReLU`.
  - Final layer outputs probabilities for three classes.
- Dropout is applied to reduce overfitting.

### 5. Evaluation
- Training stops early using the `EarlyStopping` callback if the validation loss does not improve for 10 epochs.
- A checkpoint saves the best model based on validation loss.

### 6. Deployment
- The trained model is saved as `iris_classification_model.pth`.
- You can use this saved model for inference or further fine-tuning.

---