# Step 1: Business Understanding
# The objective is to predict the profit of a startup based on several features such as R&D Spend, 
# Administration, Marketing Spend, and State. We'll use regression techniques to achieve this.

# Step 2: Data Understanding
# Importing necessary libraries to download, analyze and process the dataset

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from io import StringIO

# Fetching the dataset using the provided URL
url = "https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"

# Step 3: Data Acquisition
# Downloading the dataset using a web crawler (requests)
response = requests.get(url)
if response.status_code == 200:
    print("Dataset fetched successfully!")
    data = response.text
else:
    raise Exception("Failed to download the dataset")

# Loading the dataset into a DataFrame
df = pd.read_csv(StringIO(data))

# Output dataset summary in markdown format
print("\n## Dataset Summary:")
print(df.info())

# Displaying the first 5 rows of the dataset
print("\n## First 5 rows of the dataset:")
print(df.head())

# Step 4: Data Preparation
# Separating the features (X) and the target (y)
X = df.drop('Profit', axis=1)  # All features except the target
y = df['Profit']  # Target variable (Profit)

# One-hot encoding for the 'State' variable (converting categorical to numerical)
X = pd.get_dummies(X, columns=['State'], drop_first=True)

# Splitting the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Printing the shapes of training and test data
print(f"\n## Training Data Shape: {X_train.shape}")
print(f"## Test Data Shape: {X_test.shape}")

# Step 5: Modeling
# Initialize the Linear Regression model and train it on the training data
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Step 6: Evaluation (Linear Regression)
# Making predictions on the test data using the linear regression model
y_pred = linear_reg.predict(X_test)

# Calculating the Mean Squared Error (MSE) for linear regression
mse = mean_squared_error(y_test, y_pred)
print(f"\n## Linear Regression MSE: {mse}")

# Step 7: Feature Selection using Lasso
# Initialize the Lasso regression model for feature selection and training
lasso = Lasso(alpha=0.1)  # Adjust alpha for stronger/softer regularization
lasso.fit(X_train, y_train)

# Making predictions using the Lasso model
y_pred_lasso = lasso.predict(X_test)

# Calculating the MSE for Lasso regression
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"\n## Lasso Regression MSE: {mse_lasso}")

# Checking which features were selected by Lasso (non-zero coefficients)
lasso_coef = pd.Series(lasso.coef_, index=X_train.columns)

# Output selected features using markdown format
print("\n## Lasso Selected Features (non-zero coefficients):")
print(lasso_coef[lasso_coef != 0])

# Step 8: Final Predictions (optional, using Lasso model)
# Output the final predictions using the Lasso model in markdown format
print("\n## Final Predictions using Lasso Model:")
print(y_pred_lasso)