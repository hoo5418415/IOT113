
---
#chatgpt prompt
I want to use kaggle Titanic - Machine Learning from Disaster ，and use Hyperparameter Optimization to optimise the problem
---

# Titanic - Machine Learning from Disaster 🚢

This project is a solution to the Kaggle competition **Titanic - Machine Learning from Disaster**. The goal is to predict whether a passenger survived the Titanic disaster based on their characteristics using machine learning. This repository follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to ensure a systematic and structured approach.

---

## CRISP-DM Methodology 🛠️

### 1. **Business Understanding**
- **Objective:** Predict the survival of Titanic passengers (binary classification: `Survived` = 1 or 0).
- **Impact:** Understanding the factors affecting survival can provide insights into safety procedures and disaster management.
- **Dataset:** The dataset consists of:
  - `train.csv`: Training data (contains `Survived` label).
  - `test.csv`: Test data (used for prediction and submission).
  - [Dataset Link](https://www.kaggle.com/competitions/titanic/data)

---

### 2. **Data Understanding**
Exploratory data analysis (EDA) was performed to understand the dataset’s structure, characteristics, and relationships between variables.

- **Features Overview:**
  - `Survived`: Target variable (1 = survived, 0 = did not survive).
  - `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
  - `Sex`: Gender of the passenger.
  - `Age`: Age of the passenger.
  - `SibSp`: Number of siblings/spouses aboard.
  - `Parch`: Number of parents/children aboard.
  - `Fare`: Ticket fare.
  - `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

- **Initial Observations:**
  - Missing values in `Age`, `Cabin`, and `Embarked`.
  - Categorical variables (`Sex`, `Embarked`) require encoding.

- **Key Insights from EDA:**
  - Gender (`Sex`) is a strong predictor of survival, with females having a higher survival rate.
  - Ticket class (`Pclass`) significantly affects survival, with passengers in 1st class having the highest survival rate.
  - Family-related features like `SibSp` and `Parch` show relationships with survival when combined into a new feature `FamilySize`.

---

### 3. **Data Preparation**
Steps taken to preprocess the data for machine learning:

1. **Handling Missing Values:**
   - Imputed missing values in `Age` using the median.
   - Filled missing values in `Embarked` with the mode.
   - Dropped the `Cabin` column due to a high percentage of missing data.

2. **Feature Engineering:**
   - Created a `FamilySize` feature by combining `SibSp` and `Parch`.
   - Extracted `Title` from the passenger's name to capture social status.
   - Binned continuous features like `Age` and `Fare` into categorical ranges to address non-linearity.

3. **Encoding Categorical Variables:**
   - Converted categorical variables (`Sex`, `Embarked`, `Title`) into numerical values using one-hot encoding.

4. **Scaling Numeric Features:**
   - Scaled numeric columns such as `Age` and `Fare` to standardize the data.

5. **Data Splitting:**
   - Split the dataset into training and validation subsets using an 80:20 ratio to evaluate model performance.

---

### 4. **Modeling**
Various machine learning models were tested to predict survival. The primary focus was on binary classification models:

- Models Tested:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting Models (XGBoost, LightGBM, CatBoost)

- **Model Selection:**
  - Random Forest was chosen as the final model after hyperparameter tuning.
  - Key hyperparameters (e.g., number of trees, max depth, min samples split) were optimized using **Optuna**, a hyperparameter optimization library.

---

### 5. **Evaluation**
The selected model was evaluated using the validation dataset:

- **Metrics:**
  - Accuracy
  - Precision, Recall, and F1-Score
  - Confusion Matrix

- **Performance:**
  - Validation Accuracy: **~83%**
  - Kaggle Leaderboard Score: **~0.78**

- **Insights:**
  - Gender (`Sex`) and ticket class (`Pclass`) were the most important features influencing survival.
  - FamilySize and Title features also provided additional predictive power.

---

### 6. **Deployment**
The final model was applied to the test dataset to generate predictions for Kaggle submission:

- **Submission File:**
  - The submission file contains two columns: `PassengerId` and `Survived`.
  - Predictions were uploaded to the Kaggle platform, achieving a leaderboard score of approximately 0.78.

---

## Results 📊

- **Model Performance:**
  - Validation Accuracy: **~83%**
  - Kaggle Leaderboard Score: **~0.78**

- **Key Takeaways:**
  - Females (Sex = female) had a higher likelihood of survival.
  - Passengers in 1st class (Pclass = 1) had significantly better survival rates.
  - Traveling in larger family groups had mixed effects, with optimal survival rates for moderate family sizes.

---
