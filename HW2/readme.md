# 使用本地端執行(因colab沒有optuna

# Titanic Classification: A CRISP-DM Approach

This repository applies the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to solve the Titanic survival classification problem using Python. The objective is to predict the survival outcome of passengers based on features such as age, sex, class, and other socio-economic factors. We use `RandomForestClassifier`, feature selection (`SelectKBest`), and hyperparameter optimization with `Optuna` to achieve a robust model.

## CRISP-DM Methodology

The CRISP-DM framework includes six phases that help structure the data mining process: **Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment**.

### 1. Business Understanding

The goal of this project is to build a predictive model that can determine if a passenger survived the Titanic disaster, based on their demographic and socio-economic features. This classification task has various applications in the field of data science, providing insights into feature importance and building experience with feature engineering, model tuning, and evaluation.

### 2. Data Understanding

The Titanic dataset includes features such as `Age`, `Sex`, `Pclass` (ticket class), `Fare`, `Embarked` (port of embarkation), and more. We performed the following exploratory steps:

- Loaded the dataset and reviewed feature descriptions and distributions.
- Checked for missing values to determine how they could affect model accuracy.
- Visualized relationships between features and survival outcomes, such as survival rates by sex, class, and age, using Seaborn for basic data visualization.

### 3. Data Preparation

Data preparation includes all preprocessing steps required before training the model, covering feature selection, handling missing values, and encoding categorical data. 

#### Steps:
1. **Feature Selection**: Dropped columns that were unlikely to contribute to the model, including `PassengerId`, `Name`, `Ticket`, and `Cabin`.
2. **Separating Target and Features**: Created `X` for features and `y` for the target variable (`Survived`).
3. **Handling Missing Values**: Used `SimpleImputer` to fill missing numerical data with the mean and categorical data with the most frequent values.
4. **Encoding Categorical Variables**: Applied `OneHotEncoder` to `Sex` and `Embarked` features.
5. **Scaling Numerical Data**: Standardized numerical features to improve model convergence.

The `ColumnTransformer` and `Pipeline` from `scikit-learn` were used to streamline these transformations, ensuring consistency across training and validation.

### 4. Modeling

We used a **RandomForestClassifier** as the primary model for this classification problem. Additionally, we incorporated **SelectKBest** for feature selection and **Optuna** for hyperparameter tuning.

#### Modeling Process:
1. Defined an **objective function** for `Optuna` to optimize the model’s performance, searching for the best `k` features and ideal hyperparameters (`n_estimators` and `max_depth`).
2. Used `SelectKBest` with `f_classif` (ANOVA F-test) to score features and select the top `k` features, optimized by `Optuna`.
3. Ran multiple trials with `Optuna` to determine the best combination of features and hyperparameters that maximize validation accuracy.

### 5. Evaluation

With the optimized pipeline, we evaluated the model's performance on the validation set using the following metrics:
- **Accuracy**: The percentage of correctly classified samples.
- **Confusion Matrix**: To analyze the model's performance in terms of true/false positives and negatives.
- **Classification Report**: Includes precision, recall, and F1-score for each class, offering a detailed view of model performance.

The final model and feature selection were chosen based on the best validation accuracy achieved during the `Optuna` trials.

### 6. Deployment

To generate predictions on the test set, we used the final optimized model with the best parameters found by `Optuna`. We saved the predictions in `submission.csv`, including `PassengerId` and `Survived` status for each test passenger, in a format compatible with Kaggle’s Titanic competition submission guidelines.